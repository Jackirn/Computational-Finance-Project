clearvars; close all; clc;
rng(42); % Seed

% Paths
baseDir = fileparts(mfilename('fullpath'));
addpath(fullfile(baseDir, 'plots & prints'));
addpath(fullfile(baseDir, 'utils'));

%% Data Loading & Initial Setup
% Prices, returns, moments and base constraints 
[data, constr] = setup_data();
NumAssets = data.NumAssets;

%% Point 1.a: Standard Efficient Frontier (Mean-Variance)
fprintf('Computing Standard Frontier...\n');
% Compute frontier with basic constraints (Def <= 45%, Neu >= 20%, ub=30%)
[Frontier_MV, idx_MVP, idx_MSRP] = solve_efficient_frontier(data.ExpRet, data.V, constr, 100);

% Output & Plot
w_MVP  = Frontier_MV.Weights(:, idx_MVP);
w_MSRP = Frontier_MV.Weights(:, idx_MSRP);
Print_Ptfs(Frontier_MV.Ret(idx_MVP),  Frontier_MV.Vol(idx_MVP),  w_MVP,  'A (MVP)',  0, Frontier_MV.Sharpe(idx_MSRP));
Print_Ptfs(Frontier_MV.Ret(idx_MSRP), Frontier_MV.Vol(idx_MSRP), w_MSRP, 'B (MSRP)', 1, Frontier_MV.Sharpe(idx_MSRP));
Plot_Frontier(Frontier_MV.Vol, Frontier_MV.Ret, NumAssets, data.V, data.ExpRet, Frontier_MV.Sharpe);

%% Point 1.b: Robust Frontier (Resampling)
fprintf('Computing Robust Frontier...\n');
[RobustRes] = solve_robust_frontier(data, constr);

% Output & Plot
Print_Ptfs(RobustRes.ret_MVP,  RobustRes.vol_MVP,  RobustRes.w_MVP,  'C (Robust MVP)',  0, RobustRes.SR);
Print_Ptfs(RobustRes.ret_MSRP, RobustRes.vol_MSRP, RobustRes.w_MSRP, 'D (Robust MSRP)', 1, RobustRes.SR);
Plot_Robust_Frontier(RobustRes.Risk, RobustRes.Ret, NumAssets, data.V, data.ExpRet, RobustRes.w_MVP, RobustRes.w_MSRP, constr.rf);
Plot_Both_Frontiers(Frontier_MV.Vol, Frontier_MV.Ret, RobustRes.Risk, RobustRes.Ret, w_MVP, w_MSRP, RobustRes.w_MVP, RobustRes.w_MSRP, data.V, data.ExpRet, constr.rf);

%% Point 2: Black-Litterman
fprintf('Computing Black-Litterman...\n');
[mu_BL, ~] = solve_black_litterman(data, 3); % 3 views

constr_BL = constr; 
constr_BL.ub = ones(NumAssets, 1);     % no ub
constr_BL.b(end-NumAssets+1:end) = 1;  % update b for new ub

% Recompute frontier with new constraints
[Frontier_BL, idx_MVP_BL, idx_MSRP_BL] = solve_efficient_frontier(mu_BL', data.V, constr_BL, 100);

% Output
w_MVP_BL  = Frontier_BL.Weights(:, idx_MVP_BL);
w_MSRP_BL = Frontier_BL.Weights(:, idx_MSRP_BL);
Print_Ptfs(Frontier_BL.Ret(idx_MVP_BL), Frontier_BL.Vol(idx_MVP_BL), w_MVP_BL, 'E (BL MVP)', 0, Frontier_BL.Sharpe(idx_MSRP_BL));
Print_Ptfs(Frontier_BL.Ret(idx_MSRP_BL), Frontier_BL.Vol(idx_MSRP_BL), w_MSRP_BL, 'F (BL MSRP)', 1, Frontier_BL.Sharpe(idx_MSRP_BL));

%% Point 3: Alternative Risk (MDR & Entropy)
fprintf('Computing Alternative Risk Portfolios...\n');

% New constraints
constr_Alt = update_constraints_point3(constr, data.groups);

[AltRes] = solve_alternative_risk(data, constr_Alt);

% Output MDR
Print_MDR_Ptf(AltRes.w_MDR, AltRes.DR_vals, AltRes.target_Vol, AltRes.idx_MDR, data.ExpRet);
Plot_DR_Frontier(AltRes.target_Vol, AltRes.DR_vals);

% Output Entropy
Print_ME_Ptf(AltRes.w_ME', AltRes.Ent_vals, AltRes.target_Vol, AltRes.idx_ME, data.ExpRet);
Plot_Entropy_Frontier(AltRes.target_Vol, AltRes.Ent_vals);

Print_Portfolio_Comparison(AltRes.Metrics.DR_eq, AltRes.Metrics.DR_MDR, AltRes.Metrics.DR_ME, ...
                           AltRes.Metrics.Vol_eq, AltRes.Metrics.Vol_MDR, AltRes.Metrics.Vol_ME, ...
                           AltRes.Metrics.SR_eq, AltRes.Metrics.SR_MDR, AltRes.Metrics.SR_ME, ...
                           AltRes.Metrics.HI_eq, AltRes.Metrics.HI_MDR, AltRes.Metrics.HI_ME);

%% Point 4: PCA & CVaR
fprintf('Computing PCA and CVaR Optimization...\n');
[PCA_Res, CVaR_Res] = solve_pca_cvar(data, constr);

% Output PCA
fprintf('Selected k=%d factors. Total Explained Variance: %.2f%%\n', PCA_Res.k, sum(PCA_Res.ExplainedVar)*100);
Plot_PCA_Variance(PCA_Res.ExplainedVar);
Plot_PCA_Cumulative(PCA_Res.n_pc, PCA_Res.CumExplVar);

% Output Ptfs I-J
Print_Sharpe_PCA(CVaR_Res.w_sharpe, CVaR_Res.fval_sharpe, CVaR_Res.CovarPCA, data.ExpRet);
Print_Portfolio_J(CVaR_Res.min_vol_j, 0.10, CVaR_Res.exitflag, CVaR_Res.output, ...
                  CVaR_Res.vol_ottenuta, CVaR_Res.portfolio_cvar, CVaR_Res.returns_portfolio, CVaR_Res.w_cvar);