function [data, constr] = setup_data()
% SETUP_DATA Loads financial data and initializes base constraints.
%
%   [data, constr] = SETUP_DATA() reads asset prices, market capitalization,
%   and mapping tables from CSV files. It calculates financial moments 
%   (Expected Returns, Covariance) and sets up the initial constraints 
%   for Point 1 of the project.
%
%   OUTPUTS:
%     data   - Struct containing:
%              .prices    (Asset prices)
%              .dates     (Date vector)
%              .names     (Asset names)
%              .logret    (Logarithmic returns)
%              .ExpRet    (Annualized Expected Returns)
%              .V         (Annualized Covariance Matrix)
%              .w_MKT     (Market Capitalization Weights)
%              .groups    (Struct with logical vectors: Defensive, Neutral, Cyclical)
%     constr - Struct containing constraints for Point 1:
%              .A, .b     (Linear Inequality Constraints)
%              .Aeq, .beq (Linear Equality Constraints)
%              .lb, .ub   (Lower and Upper bounds)
%              .rf        (Risk-free rate, set to 0)

    % Get current file path to locate CSV folder dynamically
    baseDir = fileparts(mfilename('fullpath'));
    csv_path = fullfile(baseDir, '..', 'csv'); 
    
    % Load Prices & Dates 
    t_p = readtable(fullfile(csv_path, 'asset_prices.csv'));
    
    % Extract dates (First column)
    % We use standard table indexing to avoid 'Time' property issues
    dt = t_p{:,1}; 
    
    % Filter Dates (In-Sample Period)
    start_dt = datetime('02/01/2018', 'InputFormat', 'dd/MM/yyyy');
    end_dt   = datetime('30/12/2022', 'InputFormat', 'dd/MM/yyyy');
    rng_date = (dt >= start_dt) & (dt <= end_dt);
    
    subsample = t_p(rng_date, :);
    
    % Populate Data Structure
    data.prices = subsample{:, 2:end}; 
    data.dates  = subsample{:, 1};     
    data.names  = subsample.Properties.VariableNames(2:end);
    
    % Calculate Returns & Moments
    ret = data.prices(2:end,:)./data.prices(1:end-1,:);
    data.logret = log(ret);
    data.ExpRet = mean(data.logret)*252; % Annualized Mean
    data.V = cov(data.logret)*252;       % Annualized Covariance
    data.NumAssets = width(data.logret);
    
    % Market Capitalization Weights 
    t_cw = readtable(fullfile(csv_path, 'capitalization_weights.csv'));
    vals = t_cw{:, 3};
    % Normalize to sum to 1
    data.w_MKT = vals(1:data.NumAssets) ./ sum(vals(1:data.NumAssets));
    
    % Asset Grouping 
    % Load Mapping Table
    data.table_map = readtable(fullfile(csv_path, 'mapping_table.csv'));
    
    % Store groups as Column Vectors (Nx1)
    % IMPORTANT: We do NOT transpose here. They must remain columns 
    % to allow correct transposition later in update_constraints functions.
    data.groups.defensive = contains(data.table_map.MacroGroup, 'Defensive'); 
    data.groups.neutral   = contains(data.table_map.MacroGroup, 'Neutral');
    data.groups.cyclical  = contains(data.table_map.MacroGroup, 'Cyclical');
    
    % Base Constraints (Point 1 Configuration) 
    constr.lb = zeros(data.NumAssets, 1);
    constr.ub = 0.30 * ones(data.NumAssets, 1); % Max 30% per asset
    constr.rf = 0;
    
    % Construct Group Matrices for Point 1
    % Matrix G maps assets to groups (rows = groups, cols = assets)
    G = zeros(2, data.NumAssets);
    G(1, data.groups.defensive) = 1; 
    G(2, data.groups.neutral)   = 1;
    
    UpperGroup = [0.45; 1.00]; % Defensive <= 45%
    LowerGroup = [0; 0.20];    % Neutral >= 20%
    
    % Linear Inequality Construction (A*x <= b)
    % Row 1:  sum(Def) <= 0.45
    % Row 2: -sum(Neu) <= -0.20 (converted from >= 0.20)
    % Rows 3..N+2: Identity matrix to enforce individual UB as linear constraints
    %              (Redundant with 'ub' but useful for some solvers/plots)
    constr.A = [G(1,:); -G(2,:); eye(data.NumAssets)];
    constr.b = [UpperGroup(1); -LowerGroup(2); constr.ub];
    
    % Linear Equality Construction (sum(w) = 1)
    constr.Aeq = ones(1, data.NumAssets);
    constr.beq = 1;
end