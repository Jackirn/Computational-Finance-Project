function [PCA_Res, CVaR_Res] = solve_pca_cvar(data, constr)
% SOLVE_PCA_CVAR Performs PCA Factor Analysis and CVaR Optimization.
%
%   [PCA_Res, CVaR_Res] = SOLVE_PCA_CVAR(data, constr) executes two main tasks
%   corresponding to Exercise 4 (Version C):
%
%   1. PCA Analysis:
%      - Standardizes returns and performs Principal Component Analysis.
%      - Selects k factors to explain at least 80% of the variance.
%      - Reconstructs the asset covariance matrix using a PCA factor model:
%        Sigma_PCA = Lambda * (L * F * L' + D) * Lambda
%
%   2. Optimization (Portfolios I and J):
%      - Portfolio I: Maximum Sharpe Ratio using the PCA-reconstructed covariance,
%        under standard constraints (sum w = 1, 0 <= w <= 0.25) and
%        a volatility cap of 14% annualized.
%      - Portfolio J: Minimum Historical CVaR (95%) with a target volatility constraint
%        of 10% annualized under the same standard constraints.
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

    % PCA Analysis 

    % Standardize Log Returns (daily)
    muR = mean(data.logret);     % 1 x N (daily mean returns)
    sigmaR = std(data.logret);   % 1 x N (daily volatilities)
    RetStd = (data.logret - muR) ./ sigmaR;

    % Perform PCA on standardized returns (equivalent to correlation matrix PCA)
    [Loadings_All, Scores_All, latent, ~, explained] = pca(RetStd);

    % Determine k factors to reach at least 80% cumulative explained variance
    cumVar = cumsum(latent) / sum(latent);
    k = find(cumVar >= 0.80, 1, 'first');

    % Store PCA stats
    PCA_Res.k            = k;
    PCA_Res.ExplainedVar = latent(1:k) / sum(latent);
    PCA_Res.CumExplVar   = cumsum(explained);
    PCA_Res.n_pc         = 1:length(explained);

    % Factor Model Reconstruction
    Loadings    = Loadings_All(:, 1:k);          % N x k
    covarFactor = cov(Scores_All(:, 1:k));       % k x k

    % Idiosyncratic variance in standardized space
    epsilon = RetStd - Scores_All(:, 1:k) * Loadings';
    Psi_std = var(epsilon, 0, 1);               % 1 x N

    % Reconstruct Covariance Matrix via PCA Factor Model
    % Sigma_PCA = Lambda * (L * Cov_factors * L' + D_resid) * Lambda
    Lambda = diag(sigmaR);                      % D_vol
    D      = diag(Psi_std);                     % D_resid (std space)
    CVaR_Res.CovarPCA = Lambda * (Loadings * covarFactor * Loadings' + D) * Lambda;

    % Optimization Setup 

    x0      = ones(data.NumAssets, 1) / data.NumAssets;   % start from equally weighted
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

    % Local Constraints for Exercise 4 (standard constraints)
    lb_local = zeros(data.NumAssets, 1);
    ub_local = 0.25 * ones(data.NumAssets, 1);

    % Portfolio I: Max Sharpe (PCA Covariance, vol cap 14%)

    % Objective: minimize negative Sharpe Ratio
    % (Sharpe computed on daily returns, consistent with daily covariance)
    func_sharpe = @(w) - ( (muR * w) / sqrt(w' * CVaR_Res.CovarPCA * w) );

    % Volatility cap: 14% annualized
    target_vol_ann_I   = 0.14;
    target_vol_daily_I = target_vol_ann_I / sqrt(252);

    % Non-linear inequality constraint: vol(w) <= target_vol_daily_I
    nonlcon_sharpe = @(w) deal( ...
        sqrt(w' * CVaR_Res.CovarPCA * w) - target_vol_daily_I, ... % c(w) <= 0
        [] ...                                                      % no equality constraint
    );

    % fmincon for Portfolio I (Max Sharpe with vol cap)
    [w_s, fval_s] = fmincon( ...
        func_sharpe, x0, ...
        [], [], ...                             % no additional linear inequality
        constr.Aeq, constr.beq, ...             % sum w = 1
        lb_local, ub_local, ...                 % 0 <= w <= 0.25
        nonlcon_sharpe, ...                     % vol cap 14%
        options);

    CVaR_Res.w_sharpe     = w_s;
    CVaR_Res.fval_sharpe  = fval_s;  % è negativo; la funzione di print gestisce il segno

    % Portfolio J: Min CVaR with Target Volatility 10% (standard constraints)

    alpha           = 0.05;
    target_vol_ann  = 0.10;
    target_vol_daily = target_vol_ann / sqrt(252);

    % Objective: Minimize Historical CVaR (using sample covariance / returns)
    func_cvar = @(w) compute_historical_cvar(w, data.logret, alpha);

    % Non-linear equality constraint: vol(w) == target_vol_daily
    nonlcon = @(w) deal( ...
        [], ...                                                   % no inequality constraint
        sqrt(w' * cov(data.logret) * w) - target_vol_daily ...    % ceq(w) = 0
    );

    % Feasibility Check: Global Minimum Variance (GMV) under same constraints
    fun_minvar = @(w) w' * cov(data.logret) * w;
    [~, min_var_daily] = fmincon(fun_minvar, x0, ...
                                 [], [], ...
                                 constr.Aeq, constr.beq, ...
                                 lb_local, ub_local, ...
                                 [], options);

    % Annualized minimum volatility for comparison
    CVaR_Res.min_vol_j = sqrt(min_var_daily) * sqrt(252);

    % Optimize Portfolio J (Min CVaR)
    [w_cv, fval_cv, flag, out] = fmincon( ...
        func_cvar, x0, ...
        [], [], ...
        constr.Aeq, constr.beq, ...
        lb_local, ub_local, ...
        nonlcon, ...
        options);

    % Store results
    CVaR_Res.w_cvar         = w_cv;
    CVaR_Res.portfolio_cvar = fval_cv;
    CVaR_Res.exitflag       = flag;
    CVaR_Res.output         = out;

    % Metrics for verification / printing
    CVaR_Res.vol_ottenuta     = sqrt(w_cv' * cov(data.logret) * w_cv) * sqrt(252);
    CVaR_Res.returns_portfolio = data.logret * w_cv;
end