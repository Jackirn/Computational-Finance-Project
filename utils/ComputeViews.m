function [P,q] = ComputeViews(v,NumAssets,table)
% COMPUTEVIEWS  Build the P and q matrices for Black–Litterman investor views.
%
%   [P, q] = COMPUTEVIEWS(v, NumAssets)
%
%   INPUT:
%       v          - Number of investor views (here v = 3).
%       NumAssets  - Total number of assets in the universe.
%
%   OUTPUT:
%       P   - (v × NumAssets) matrix. Each row maps a view to the assets
%             involved. Positive coefficients = outperformers;
%             negative coefficients = underperformers.
%
%       q   - (v × 1) vector of expected returns implied by each view.
%             Units must be consistent (annualized excess returns).
%
%   DESCRIPTION:
%       This function constructs the view matrix P and the view returns q
%       used in the Black–Litterman posterior estimation.
%
%       View 1: Cyclical assets expected to outperform Neutral assets by +2%.
%       View 2: Asset 10 expected to underperform the Defensive group by –0.7%.
%       View 3: Asset 2 expected to outperform Asset 13 by +1%.
%
%   NOTES:
%       - The function relies on the workspace variable "table" containing
%         the macro-group classification of each asset.
%       - Views involving groups use equal-weighted exposure within each group.
%

    P = zeros(v, NumAssets);  
    q = zeros(v, 1);          % expected returns from views
    
    Asset_Macro = string(table.MacroGroup);
    
    % View 1: we expect Cyclical assets to outperform Neutral ones by 2% annualized
    cyclical_mask = strcmp(Asset_Macro, 'Cyclical');
    neutral_mask = strcmp(Asset_Macro, 'Neutral');
    
    c = sum(cyclical_mask);
    n = sum(neutral_mask);
    
    P(1, cyclical_mask) = 1/c;
    P(1, neutral_mask) = -1/n;
    q(1) = 0.02;
    
    % View 2: we expect Asset_10 to underperform average Defensive group by -0.7% annualized
    defensive_mask = strcmp(Asset_Macro, 'Defensive');
    
    d = sum(defensive_mask);
    
    P(2, 10) = 1;  % Asset_10
    P(2, defensive_mask) = P(2, defensive_mask) - 1/d;
    q(2) = -0.007; % underperformance
    
    % View 3: we expect Asset_2 to outperform Asset_13 by 1% annualized
    P(3, 2) = 1;   % Asset_2
    P(3, 13) = -1; % Asset_13  
    q(3) = 0.01;

end