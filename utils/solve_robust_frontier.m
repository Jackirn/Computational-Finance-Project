function [Res] = solve_robust_frontier(data, constr)
% SOLVE_ROBUST_FRONTIER Computes a Robust Efficient Frontier via Resampling.
%
%   Res = SOLVE_ROBUST_FRONTIER(data, constr) performs a Monte Carlo 
%   resampling simulation (Michaud Resampling) to construct a robust 
%   efficient frontier. It simulates N scenarios of market parameters, 
%   computes optimal frontiers for each, and averages the portfolio weights.
%
%   The function applies the following constraints (Hardcoded for Point 1.b):
%     1. Long-only, Fully Invested.
%     2. Individual Asset Bounds: [0, 30%].
%     3. Group Constraints: 
%          - Defensive Assets <= 45%
%          - Neutral Assets   >= 20%
%
%   INPUTS:
%     data   - Struct containing:
%              .names (Asset names)
%              .ExpRet (Expected Returns vector)
%              .V (Covariance Matrix)
%              .groups (Logical vectors for Defensive/Neutral/Cyclical)
%     constr - Struct containing:
%              .rf (Risk-free rate)
%
%   OUTPUT:
%     Res    - Struct containing robust frontier data:
%              .Risk, .Ret (Vectors of frontier points)
%              .w_MVP, .vol_MVP, .ret_MVP (Robust Min Variance Portfolio)
%              .w_MSRP, .vol_MSRP, .ret_MSRP, .SR (Robust Max Sharpe Portfolio)

    % Portfolio Object Setup
    p = Portfolio('AssetList', data.names);
    p = setDefaultConstraints(p); % Standard constraints: sum(w)=1, w>=0
    p = setBounds(p, 0, 0.30);    % Individual asset bounds: [0%, 30%]
    
    % Group Constraints Setup
    % Matrix G maps assets to groups for the Portfolio object
    G = zeros(2, data.NumAssets);
    G(1, data.groups.defensive) = 1;
    G(2, data.groups.neutral) = 1;
    
    % Define Limits: Defensive <= 45%, Neutral >= 20%
    LowerGroup = [0; 0.20]; 
    UpperGroup = [0.45; 1.00];
    p = setGroups(p, G, LowerGroup, UpperGroup);
    
    % Monte Carlo Resampling Simulation
    N_sim = 200;            % Number of simulations
    nPort = 100;            % Number of points on the frontier
    Weights = zeros(data.NumAssets, nPort, N_sim);
    
    for n = 1:N_sim
        % Simulate Returns (Multivariate Normal) and Covariance (Inverse Wishart)
        % to account for estimation error.
        R_sim  = mvnrnd(data.ExpRet, data.V)';
        Cov_sim = iwishrnd(data.V, data.NumAssets);
        
        % Create a temporary portfolio with simulated moments
        P_sim = setAssetMoments(p, R_sim', Cov_sim);
        
        % Calculate efficient frontier for this simulation
        Weights(:,:,n) = estimateFrontier(P_sim, nPort);
    end
    
    % Aggregate Results (Michaud Resampling)
    % Compute the average weights across all simulated frontiers
    meanWeights = mean(Weights, 3);
    
    % Evaluate Robust Frontier
    % Calculate Risk and Return of the *averaged weights* using the *original* moments
    P_avg = setAssetMoments(p, data.ExpRet, data.V);
    [RobustRisk, RobustRet] = estimatePortMoments(P_avg, meanWeights);
    
    % Remove NaN values (infeasible points)
    valid = ~isnan(RobustRisk);
    Res.Risk = RobustRisk(valid);
    Res.Ret  = RobustRet(valid);
    meanWeights = meanWeights(:, valid);
    
    % Identify Key Portfolios on Robust Frontier
    
    % Robust Minimum Variance Portfolio (MVP)
    [Res.vol_MVP, idx_MVP] = min(Res.Risk);
    Res.ret_MVP = Res.Ret(idx_MVP);
    Res.w_MVP   = meanWeights(:, idx_MVP);
    
    % Robust Maximum Sharpe Ratio Portfolio (MSRP)
    Res.Sharpe = (Res.Ret - constr.rf) ./ Res.Risk;
    [~, idx_MSRP] = max(Res.Sharpe);
    Res.vol_MSRP = Res.Risk(idx_MSRP);
    Res.ret_MSRP = Res.Ret(idx_MSRP);
    Res.w_MSRP   = meanWeights(:, idx_MSRP);
    Res.SR = Res.Sharpe(idx_MSRP);
end