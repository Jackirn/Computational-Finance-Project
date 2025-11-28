function [Res] = solve_alternative_risk(data, constr)
% SOLVE_ALTERNATIVE_RISK Computes MDR and Maximum Entropy Frontiers.
%
%   Res = SOLVE_ALTERNATIVE_RISK(data, constr) calculates portfolios based on
%   alternative risk measures: Diversification Ratio (DR) and Entropy of 
%   Risk Contributions (ERC).
%
%   The function performs the following steps:
%     1. Feasible Volatility Range: Mathematically determines the exact
%        minimum and maximum volatility achievable under the provided 
%        constraints (which differ from Point 1).
%     2. MDR Frontier: Maximizes the Diversification Ratio at each volatility
%        target across the range.
%     3. Entropy Frontier: Maximizes the Entropy of risk contributions at
%        each volatility target (ERC optimization).
%     4. Metrics Comparison: Calculates DR, Volatility, Sharpe Ratio, and
%        Herfindahl Index for MDR, Entropy, and Equal-Weighted portfolios.
%
%   INPUTS:
%     data   - Struct containing .V (Covariance), .NumAssets, .ExpRet.
%     constr - Struct containing constraints specific to Point 3
%              (e.g., Cyclical/Defensive limits, Asset UB = 0.25).
%
%   OUTPUTS:
%     Res    - Struct containing:
%              .w_MDR, .DR_vals    (MDR Frontier data)
%              .w_ME,  .Ent_vals   (Entropy Frontier data)
%              .target_Vol         (Volatility grid used)
%              .idx_MDR, .idx_ME   (Indices of best portfolios)
%              .Metrics            (Comparison struct: DR, Vol, SR, HI)

    NumAssets = data.NumAssets;
    sigma_i = sqrt(diag(data.V)); % Vector of individual asset volatilities
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    x0 = ones(NumAssets,1)/NumAssets;
    
    % Determine Feasible Volatility Range 
    % Instead of guessing a range (e.g., 0.05 - 0.30), we calculate the 
    % exact mathematical limits imposed by the Point 3 constraints.
    
    % Find Minimum Feasible Volatility
    fun_var = @(w) w' * data.V * w;
    [~, min_var_val] = fmincon(fun_var, x0, constr.A, constr.b, constr.Aeq, constr.beq, constr.lb, constr.ub, [], options);
    min_vol = sqrt(min_var_val);
    
    % Find Maximum Feasible Volatility
    % (Maximize variance <=> Minimize negative variance)
    fun_max_var = @(w) -w' * data.V * w;
    [~, max_var_val] = fmincon(fun_max_var, x0, constr.A, constr.b, constr.Aeq, constr.beq, constr.lb, constr.ub, [], options);
    max_vol = sqrt(-max_var_val);
    
    % Create the grid exactly over this feasible range
    % Adding a small epsilon to avoid numerical instability at the boundaries
    target_Vol = linspace(min_vol * 1.0001, max_vol * 0.9999, 100);
    
    % MDR Frontier (Maximum Diversification Ratio) 
    % Objective: Minimize negative DR
    fun_DR = @(w) - (w' * sigma_i) / sqrt(w' * data.V * w);
    
    DR_vals = zeros(size(target_Vol));
    w_DR = zeros(NumAssets, length(target_Vol));
    
    for i = 1:length(target_Vol)
        sigma_tar = target_Vol(i);
        % Non-linear constraint: Portfolio Volatility == Target
        nonlcon = @(w) deal([], sqrt(w'*data.V*w) - sigma_tar);
        
        [w, fval, flag] = fmincon(fun_DR, x0, constr.A, constr.b, constr.Aeq, constr.beq, constr.lb, constr.ub, nonlcon, options);
        
        if flag > 0
            w_DR(:,i) = w; 
            DR_vals(i) = -fval; % Revert sign for storage
        else
            w_DR(:,i) = NaN;
            DR_vals(i) = NaN; 
        end
    end
    
    % Identify the portfolio with the Global Maximum DR
    [~, idx_MDR] = max(DR_vals);
    Res.w_MDR = w_DR(:, idx_MDR);
    Res.DR_vals = DR_vals;
    Res.target_Vol = target_Vol;
    Res.idx_MDR = idx_MDR;
    
    % Entropy Frontier (Maximum Entropy of Risk Contributions) 
    % k_fun calculates the normalized risk contributions of each asset
    k_fun = @(w) (w .* (data.V*w)) ./ (w' * data.V * w);
    eps_ = 1e-12; % Epsilon to avoid log(0) errors
    
    % Objective: Minimize negative Entropy (Sum(k * log(k)))
    fun_ENT = @(w) sum( k_fun(w) .* log( k_fun(w) + eps_ ) );
    
    Ent_vals = zeros(size(target_Vol));
    w_ENT = zeros(NumAssets, length(target_Vol));
    
    for i = 1:length(target_Vol)
        sigma_tar = target_Vol(i);
        nonlcon = @(w) deal([], sqrt(w'*data.V*w) - sigma_tar);
        
        [w, fval, flag] = fmincon(fun_ENT, x0, constr.A, constr.b, constr.Aeq, constr.beq, constr.lb, constr.ub, nonlcon, options);
        
        if flag > 0
            w_ENT(:,i) = w; 
            Ent_vals(i) = -fval; % Revert sign (Entropy is negative sum)
        else
            w_ENT(:,i) = NaN;
            Ent_vals(i) = NaN;
        end
    end
    
    % Identify the portfolio with Maximum Entropy
    [~, idx_ME] = max(Ent_vals);
    Res.w_ME = w_ENT(:, idx_ME);
    Res.Ent_vals = Ent_vals;
    Res.idx_ME = idx_ME; % Store index for printing
    
    % Comparison Metrics 
    w_eq = (1/NumAssets)*ones(NumAssets,1);
    
    % Helper function to calculate all metrics at once
    calc_mets = @(w) struct('DR', (w'*sigma_i)/sqrt(w'*data.V*w), ...
                            'Vol', sqrt(w'*data.V*w), ...
                            'SR', (w'*data.ExpRet' - constr.rf)/sqrt(w'*data.V*w), ...
                            'HI', sum(w.^2));
    
    M_eq = calc_mets(w_eq);
    M_mdr = calc_mets(Res.w_MDR);
    M_me = calc_mets(Res.w_ME);
    
    % Store metrics in output structure
    Res.Metrics.DR_eq = M_eq.DR; Res.Metrics.DR_MDR = M_mdr.DR; Res.Metrics.DR_ME = M_me.DR;
    Res.Metrics.Vol_eq = M_eq.Vol; Res.Metrics.Vol_MDR = M_mdr.Vol; Res.Metrics.Vol_ME = M_me.Vol;
    Res.Metrics.SR_eq = M_eq.SR; Res.Metrics.SR_MDR = M_mdr.SR; Res.Metrics.SR_ME = M_me.SR;
    Res.Metrics.HI_eq = M_eq.HI; Res.Metrics.HI_MDR = M_mdr.HI; Res.Metrics.HI_ME = M_me.HI;
end