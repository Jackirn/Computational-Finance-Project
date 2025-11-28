function [Res, idx_MVP, idx_MSRP] = solve_efficient_frontier(ExpRet, V, constr, N_points)
% SOLVE_EFFICIENT_FRONTIER Computes the Mean-Variance Efficient Frontier.
%
%   [Res, idx_MVP, idx_MSRP] = SOLVE_EFFICIENT_FRONTIER(ExpRet, V, constr, N_points)
%   calculates the efficient frontier by solving a sequence of quadratic
%   optimization problems. 
%
%   Unlike standard grid approaches, this function first identifies the
%   exact feasible range of returns [MinVar_Ret, Max_Ret] respecting all
%   constraints (Linear, Bounds, Groups), ensuring the frontier is
%   continuous and fully feasible without gaps.
%
%   INPUTS:
%     ExpRet   - Vector of Expected Returns (1 x N).
%     V        - Covariance Matrix (N x N).
%     constr   - Struct containing constraints (A, b, Aeq, beq, lb, ub, rf).
%     N_points - Number of points to compute on the frontier.
%
%   OUTPUTS:
%     Res      - Struct containing frontier data:
%                .Vol     (Vector of Volatilities)
%                .Ret     (Vector of Returns)
%                .Weights (Matrix of weights, N x N_points)
%                .Sharpe  (Vector of Sharpe Ratios)
%     idx_MVP  - Index of the Minimum Variance Portfolio in the vectors.
%     idx_MSRP - Index of the Maximum Sharpe Ratio Portfolio in the vectors.

    NumAssets = length(ExpRet);
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    x0 = ones(NumAssets,1)/NumAssets;
    
    % Determine Feasible Frontier Boundaries 
    % Instead of guessing a return range, we calculate the absolute minimum
    % and maximum returns possible given the specific constraints.
    
    % Find Global Minimum Variance Portfolio (Left Boundary)
    fun_var = @(x) x'*V*x;
    [w_minVar, ~] = fmincon(fun_var, x0, constr.A, constr.b, constr.Aeq, constr.beq, constr.lb, constr.ub, [], options);
    r_min = w_minVar' * ExpRet';
    
    % Find Maximum Return Portfolio (Right Boundary)
    % Note: Minimize negative return to find max return
    fun_ret = @(x) -x'*ExpRet';
    [w_maxRet, ~] = fmincon(fun_ret, x0, constr.A, constr.b, constr.Aeq, constr.beq, constr.lb, constr.ub, [], options);
    r_max = w_maxRet' * ExpRet';
    
    % Compute Frontier on Feasible Grid 
    % Create target return grid exclusively within feasible bounds [r_min, r_max]
    ret_range = linspace(r_min, r_max, N_points);
    
    Res.Vol = zeros(1, N_points);
    Res.Ret = zeros(1, N_points);
    Res.Weights = zeros(NumAssets, N_points);
    
    for i = 1:N_points
        r = ret_range(i);
        
        % Augment equality constraints: Existing Aeq + Target Return
        Aeq_i = [constr.Aeq; ExpRet]; 
        beq_i = [constr.beq; r];
        
        [w, ~, exitflag] = fmincon(fun_var, x0, constr.A, constr.b, Aeq_i, beq_i, constr.lb, constr.ub, [], options);
        
        if exitflag > 0
            Res.Vol(i) = sqrt(w'*V*w);
            Res.Ret(i) = w'*ExpRet'; 
            Res.Weights(:,i) = w;
        else
            % Should not happen given Step 1, but for safety:
            Res.Vol(i) = NaN; 
            Res.Ret(i) = NaN;
        end
    end
    
    % Compute Sharpe Ratios and Key Indices 
    Res.Sharpe = (Res.Ret - constr.rf) ./ Res.Vol;
    
    [~, idx_MVP]  = min(Res.Vol);
    [~, idx_MSRP] = max(Res.Sharpe);
end