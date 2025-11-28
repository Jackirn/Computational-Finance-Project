function [mu_BL, cov_BL] = solve_black_litterman(data, v_views)
% SOLVE_BLACK_LITTERMAN Computes Posterior Returns using the Black-Litterman Model.
%
%   [mu_BL, cov_BL] = SOLVE_BLACK_LITTERMAN(data, v_views) implements the
%   standard Black-Litterman framework to combine market equilibrium returns
%   (Prior) with investor views to generate a new set of expected returns
%   and covariance (Posterior).
%
%   The process involves three main steps:
%     1. Reverse Optimization: Calculates the implied market equilibrium 
%        returns (Pi) derived from market capitalization weights.
%     2. View Construction: Defines the Pick Matrix (P), View Vector (q), 
%        and Uncertainty Matrix (Omega).
%     3. Posterior Estimation: Combines the Prior and Views using Generalized
%        Least Squares (GLS) / Bayesian updating to find mu_BL.
%
%   INPUTS:
%     data    - Struct containing:
%               .w_MKT    (Market Capitalization Weights)
%               .ExpRet   (Historical Expected Returns - unused for Equilibrium, used for stats)
%               .V        (Covariance Matrix)
%               .logret   (Historical Log Returns, used for tau sizing)
%               .NumAssets(Number of assets)
%               .table_map(Asset mapping table for views)
%               .names    (Asset names)
%     v_views - Integer specifying the number of views to compute.
%
%   OUTPUTS:
%     mu_BL   - Vector of Posterior Expected Returns (N x 1).
%     cov_BL  - Matrix of Posterior Covariance (N x N).

    % Reverse Optimization: Market Implied Returns 
    
    % Calculate Market Portfolio Moments
    ExpRet_MKT = data.w_MKT' * data.ExpRet';
    sigma2_MKT = data.w_MKT' * data.V * data.w_MKT;
    
    % Risk Aversion Coefficient (Lambda)
    % Note: Risk-free rate is hardcoded to 0 as per project requirements
    lambda = (ExpRet_MKT - 0) / sigma2_MKT; 
    
    % Implied Equilibrium Returns Vector (Pi)
    mu_MKT = lambda * data.V * data.w_MKT;
    
    % Print Equilibrium Statistics
    printBLMeq(ExpRet_MKT, sigma2_MKT, lambda, data.NumAssets, data.w_MKT, mu_MKT);
    
    % View Construction (P, q, Omega) 
    
    % Scalar tau: scaling factor for the prior covariance
    % Set to 1 / N_observations (standard practice)
    tau = 1 / size(data.logret, 1);
    
    Omega = zeros(v_views);
    
    % Generate Pick Matrix (P) and View Vector (q) via external helper function
    [P, q] = ComputeViews(v_views, data.NumAssets, data.table_map); 
    
    % Construct Diagonal Uncertainty Matrix (Omega)
    % Heuristic: Uncertainty is proportional to the variance of the view portfolios
    for i = 1:v_views
        Omega(i,i) = tau * P(i,:) * data.V * P(i,:)';
    end
    
    PrintViews(q);
    
    % Black-Litterman Posterior Estimation 
    
    % Scaled Prior Covariance
    C = tau * data.V;
    
    % Solve for Posterior Mean (mu_BL) using the GLS formation
    % Solving system: (inv(C) + P'inv(Omega)P) * mu = inv(C)*Pi + P'inv(Omega)q
    % We define A_bl and b_bl to solve A * x = b
    A_bl = (C \ eye(data.NumAssets) + P'/Omega*P);
    b_bl = (P'/Omega*q + C\mu_MKT);
    
    mu_BL = A_bl \ b_bl;
    
    % Solve for Posterior Covariance
    % Formula: M = inv( inv(C) + P'inv(Omega)P )
    % Note: Inverting A_bl is equivalent to calculating the posterior covariance matrix
    % plus the uncertainty of the mean. 
    cov_BL = inv(P'/Omega*P + inv(C));
    
    % Print Comparison: Implied vs Posterior
    PrintExpRetBLM(data.NumAssets, mu_BL, mu_MKT, data.names);
end