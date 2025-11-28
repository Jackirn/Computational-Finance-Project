function [PCA_Res, CVaR_Res] = solve_pca_cvar(data, constr)
% SOLVE_PCA_CVAR Performs PCA Factor Analysis and CVaR Optimization.
%
%   [PCA_Res, CVaR_Res] = SOLVE_PCA_CVAR(data, constr) executes two main tasks
%   corresponding to Point 4 of the project:
%
%   1. PCA Analysis:
%      - Standardizes returns and performs Principal Component Analysis.
%      - Selects k factors to explain at least 85% of the variance.
%      - Reconstructs the asset covariance matrix using the Factor Model:
%        Sigma_PCA = Lambda * (L * F * L' + D) * Lambda
%
%   2. Optimization (Portfolios I and J):
%      - Portfolio I: Maximum Sharpe Ratio using the PCA-reconstructed covariance.
%      - Portfolio J: Minimum Historical CVaR (95%) with a target volatility constraint (10%).
%
%   INPUTS:
%     data       - Struct with fields: .logret, .NumAssets.
%     constr     - Struct with equality constraints (Aeq, beq).
%
%   OUTPUTS:
%     PCA_Res    - Struct containing PCA results (Explained Variance, Loadings, etc.).
%     CVaR_Res   - Struct containing optimization results:
%                  .CovarPCA (Reconstructed Matrix)
%                  .w_sharpe (Weights for Portfolio I)
%                  .w_cvar   (Weights for Portfolio J)
%                  .min_vol_j (Minimum feasible volatility for check)

    % 1. PCA Analysis 
    
    % Standardize Log Returns
    muR = mean(data.logret);
    sigmaR = std(data.logret);
    RetStd = (data.logret - muR) ./ sigmaR;
    
    % Perform PCA
    [Loadings_All, Scores_All, latent, ~, explained] = pca(RetStd);
    
    % Determine k factors to reach 85% cumulative explained variance
    cumVar = cumsum(latent) / sum(latent);
    k = find(cumVar >= 0.85, 1, 'first');
    
    % Store PCA stats
    PCA_Res.k = k;
    PCA_Res.ExplainedVar = latent(1:k) / sum(latent);
    PCA_Res.CumExplVar = cumsum(explained);
    PCA_Res.n_pc = 1:length(explained);
    
    % Factor Model Reconstruction
    Loadings = Loadings_All(:, 1:k);
    covarFactor = cov(Scores_All(:, 1:k));
    
    % Calculate Residual (Idiosyncratic) Variance
    % epsilon = Real_Std_Ret - Explained_Std_Ret
    epsilon = RetStd - Scores_All(:, 1:k) * Loadings';
    Psi_std = var(epsilon, 0, 1);
    
    % Reconstruct Covariance Matrix via PCA Factor Model
    % Sigma = D_vol * (Loadings * Cov_factors * Loadings' + D_resid) * D_vol
    Lambda = diag(sigmaR);
    D = diag(Psi_std);
    CVaR_Res.CovarPCA = Lambda * (Loadings * covarFactor * Loadings' + D) * Lambda;
    
    % 2. Optimization Setup 
    x0 = ones(data.NumAssets, 1) / data.NumAssets;
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    
    % Local Constraints for Point 4
    % Note: The exercise specifies a max weight of 25% for these portfolios,
    % distinct from the 30% used in Point 1.
    lb_local = zeros(data.NumAssets, 1);
    ub_local = 0.25 * ones(data.NumAssets, 1);
    
    % Portfolio I: Max Sharpe (PCA Covariance) 
    % Objective: Minimize negative Sharpe Ratio using PCA Covariance
    func_sharpe = @(x) - ((muR*x) / sqrt(x' * CVaR_Res.CovarPCA * x));
    
    [w_s, fval_s] = fmincon(func_sharpe, x0, [], [], constr.Aeq, constr.beq, lb_local, ub_local, [], options);
    
    CVaR_Res.w_sharpe = w_s;
    CVaR_Res.fval_sharpe = fval_s; % Result is negative (minimization), print function handles sign.
    
    % Portfolio J: Min CVaR with Target Volatility 
    alpha = 0.05;
    target_vol_ann = 0.10;
    target_vol_daily = target_vol_ann / sqrt(252);
    
    % Objective: Minimize Historical CVaR
    func_cvar = @(w) compute_historical_cvar(w, data.logret, alpha);
    
    % Non-linear Constraint: Portfolio Volatility == Target Volatility
    % nonlcon returns [c, ceq]. We use ceq for equality constraint.
    nonlcon = @(w) deal([], sqrt(w' * cov(data.logret) * w) - target_vol_daily);
    
    % Feasibility Check: Calculate Global Minimum Variance (GMV)
    % This checks if the 10% target volatility is even mathematically possible
    % given the 25% weight constraints.
    fun_minvar = @(w) w' * cov(data.logret) * w;
    [~, min_var_daily] = fmincon(fun_minvar, x0, [], [], constr.Aeq, constr.beq, lb_local, ub_local, [], options);
    
    % Correct Annualization for print comparison (sqrt(252))
    CVaR_Res.min_vol_j = sqrt(min_var_daily) * sqrt(252); 
    
    % Execute CVaR Optimization
    [w_cv, fval_cv, flag, out] = fmincon(func_cvar, x0, [], [], constr.Aeq, constr.beq, lb_local, ub_local, nonlcon, options);
    
    % Output Results
    CVaR_Res.w_cvar = w_cv;
    CVaR_Res.portfolio_cvar = fval_cv;
    
    % Calculate obtained metrics for verification
    CVaR_Res.vol_ottenuta = sqrt(w_cv' * cov(data.logret) * w_cv) * sqrt(252);
    CVaR_Res.returns_portfolio = data.logret * w_cv;
    CVaR_Res.exitflag = flag;
    CVaR_Res.output = out;
end