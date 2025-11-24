clear 
close all
clc
rng(42)
addpath('plots & prints')
addpath('utils')

%% Data
    
baseDir = fileparts(mfilename('fullpath')); 
csv     = fullfile(baseDir, 'csv');         
addpath(csv, '');                           
path_map = [csv filesep];

% Asset Prices
filename = 'asset_prices.csv';
table_prices = readtable(strcat(path_map, filename));

dt = table_prices(:,1).Variables;
values = table_prices(:,2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm);

% Capitalization weights
filename2 = 'capitalization_weights.csv';
table_capw2 = readtable(strcat(path_map,filename2));

% Mapping Table
filename = 'mapping_table.csv';
table = readtable(strcat(path_map, filename));

defensive_assets = contains(table.MacroGroup, 'Defensive');
neutral_assets = contains(table.MacroGroup, 'Neutral'); 
cyclical_assets = contains(table.MacroGroup, 'Cyclical');

%% Selection of a subset of Dates (In-Sample Dataset)

start_dt = datetime('02/01/2018', 'InputFormat', 'dd/MM/yyyy'); 
end_dt   = datetime('30/12/2022', 'InputFormat', 'dd/MM/yyyy');

rng = timerange(start_dt, end_dt,'closed'); 
subsample = myPrice_dt(rng,:);

prices_val = subsample.Variables;
dates_ = subsample.Time;

%% Calculate returns
 
ret = prices_val(2:end,:)./prices_val(1:end-1,:);
logret = log(ret);

% Annualization: 252 trading days per year
ExpRet = mean(logret)*252;       % expected annual return
V = cov(logret)*252;             % annual covariance matrix
std_ = std(logret)*sqrt(252);    % annual volatility
NumAssets = width(logret);

%% Generate N random portfolio

N = 100000;
RetPtfs = zeros(1,N);
VolaPtfs = zeros(1,N);
SharpePtfs = zeros(1,N);

for n = 1:N
    w = rand(1,NumAssets); % vector of dim. = NumAssets of random numbers drawn from a uniform distr. in (0,1).
    w = w./sum(w); % normalize weights --> their sum has to be = 1.

    RetPtfs(n) = w*ExpRet';
    VolaPtfs(n) = sqrt(w*V*w');
    SharpePtfs(n) = RetPtfs(n)/VolaPtfs(n);
end

%% Standard Constraints & Global Variables and functions

% Bound constraints
lb = zeros(NumAssets,1);
ub = 0.30*ones(NumAssets,1);

% Group membership vectors
isDef = defensive_assets;   % logical indices
isNeu = neutral_assets;

% Macro groups matrices: same structure for fmincon and portfolio object
G = zeros(2, NumAssets);
G(1, isDef) = 1;      % Defensive
G(2, isNeu) = 1;      % Neutral

LowerGroup = [0; 0.20];   % Defensive ≥ 0, Neutral ≥ 20%
UpperGroup = [0.45; 1.00];% Defensive ≤45%, Neutral ≤100%

% Convert group constraints into fmincon A,b form

% Defensive ≤ 45%  -> sum(def) ≤ 0.45
A_def = G(1,:);
b_def = UpperGroup(1);

% Neutral ≥ 20% → -sum(neu) ≤ -0.20
A_neu = -G(2,:);
b_neu = -LowerGroup(2);

% Combine all linear constraints
A_ineq = [A_def; A_neu; eye(NumAssets)];
b_ineq = [b_def; b_neu; ub];

x0 = ones(NumAssets,1)/NumAssets;      % initial guess = equal weights
rf = 0;
fun = @(x) x'*V*x;                     % objective = variance
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

%% Point 1.a) Compute the Efficient Frontier

ret_range = linspace(min(RetPtfs), max(RetPtfs),100);

FrontierVola = zeros(1,length(ret_range)); % we have to compute the exp_ret and volat for every point in the ret_range.
FrontierRet  = zeros(1,length(ret_range));

WeightsFrontier = zeros(16,length(ret_range));
feasible_points = true(1, length(ret_range));

for i = 1:length(ret_range)
    r = ret_range(i);
    Aeq = [ones(1,NumAssets); ExpRet]; % added constraints: sum weights =1, target return
    beq = [1; r];

    [w_opt, ~, exitflag] = fmincon(fun, x0, A_ineq, b_ineq, Aeq, beq, lb, ub,[], options); 

    if exitflag > 0  % Soluzione trovata
        FrontierVola(i) = sqrt(w_opt'*V*w_opt);
        FrontierRet(i)  = w_opt'*ExpRet';
        WeightsFrontier(:,i) = w_opt;
    else  % Punto infattibile
        feasible_points(i) = false;
        FrontierVola(i) = NaN;
        FrontierRet(i) = NaN;
    end
end

% MVP with the above constraints

[~, idx_MVP] = min(FrontierVola);
w_MVP = WeightsFrontier(:, idx_MVP);
vol_MVP = FrontierVola(idx_MVP);
ret_MVP = FrontierRet(idx_MVP);

% MSRP with the above constraints

SharpeFrontier = (FrontierRet - rf) ./ FrontierVola;
[~, idx_MSRP] = max(SharpeFrontier);
w_MSRP = WeightsFrontier(:, idx_MSRP);
vol_MSRP = FrontierVola(idx_MSRP);
ret_MSRP = FrontierRet(idx_MSRP);

% Print and Plot
Print_Ptfs(ret_MVP, vol_MVP, w_MVP, 'A (MVP)')
Print_Ptfs(ret_MSRP, vol_MSRP, w_MSRP, 'B (MSRP)')
Plot_Frontier(FrontierVola,FrontierRet,NumAssets,V,ExpRet,SharpeFrontier)

%% 1.b Robust Frontier

p = Portfolio('AssetList', nm);    
p = setDefaultConstraints(p);          % weight sum = 1, w >= 0

p = setBounds(p, p.LowerBound, ub);    % LowerBound already taken into 
                                       % account by setDefaultConstraints
                  
p = setGroups(p, G, LowerGroup, UpperGroup);

N = 200;               % number of simulations
nAssets = p.NumAssets;
nPort = 100;           % points on each frontier

RiskPtfSim = zeros(nPort, N);
RetPtfSim  = zeros(nPort, N);
Weights    = zeros(nAssets, nPort, N); % store weights for each simulation

for n = 1:N
    
    % Simulate a vector of Expected Returns and a Covariance matrix
    R_sim  = mvnrnd(ExpRet, V)';
    Cov_sim = iwishrnd(V, nAssets);
    
    % Set simulated moments
    P_sim = setAssetMoments(p, R_sim', Cov_sim); 
    
    % Estimates Efficient Frontier with estimated paramenters
    w_sim = estimateFrontier(P_sim, nPort);     % nAssets x nPort
    Weights(:,:,n) = w_sim;
    
    % Ptf moments on simulated frontier
    [pf_risk, pf_ret] = estimatePortMoments(P_sim, w_sim);
    RiskPtfSim(:,n) = pf_risk;
    RetPtfSim(:,n)  = pf_ret;
end 

% Mean weights on every point of the frontier
meanWeights = mean(Weights,3);

P_avg = setAssetMoments(p, ExpRet, V); 
[RobustRisk, RobustRet] = estimatePortMoments(P_avg, meanWeights);

% For safety take out NaN
valid_points = ~isnan(RobustRisk) & ~isnan(RobustRet);
RobustRisk   = RobustRisk(valid_points);
RobustRet    = RobustRet(valid_points);
meanWeights  = meanWeights(:, valid_points);

% Robust MVP (C): punto con varianza minima sulla frontiera robusta
[vol_MVP_RF, idx_MVP_RF] = min(RobustRisk);
ret_MVP_RF = RobustRet(idx_MVP_RF);
w_MVP_RF   = meanWeights(:, idx_MVP_RF);

Print_Ptfs(ret_MVP_RF, vol_MVP_RF, w_MVP_RF, 'C (Robust MVP)');

% Robust MSRP (D): punto con massimo Sharpe rispetto a rf
SharpeRatios_RF = (RobustRet - rf) ./ RobustRisk;
[~, idx_MSRP_RF] = max(SharpeRatios_RF);
w_MSRP_RF   = meanWeights(:, idx_MSRP_RF);
vol_MSRP_RF = RobustRisk(idx_MSRP_RF);
ret_MSRP_RF = RobustRet(idx_MSRP_RF);

% Print and Plots
Print_Ptfs(ret_MSRP_RF, vol_MSRP_RF, w_MSRP_RF, 'D (Robust MSRP)');
Plot_Robust_Frontier(RobustRisk, RobustRet, NumAssets, V, ExpRet, ...
                     w_MVP_RF, w_MSRP_RF, rf);
Plot_Both_Frontiers(FrontierVola, FrontierRet, RobustRisk, RobustRet, ...
                    w_MVP, w_MSRP, w_MVP_RF, w_MSRP_RF, V, ExpRet, rf);

%% Point 2.a) BLM equilibrium returns

cap_weights = table_capw2{:, 3};
w_MKT = cap_weights(1:NumAssets) ./ sum(cap_weights(1:NumAssets)); % market weights

% Calculate market portfolio statistics
ExpRet_MKT = w_MKT' * ExpRet';   % Expected return of market portfolio
sigma2_MKT = w_MKT' * V * w_MKT; % Variance of market portfolio

lambda = (ExpRet_MKT - rf) / sigma2_MKT; % Common approach for computing lambda
mu_MKT = lambda * V * w_MKT;             % Implied Equilibrium Return Vector

% Print
printBLMeq(ExpRet_MKT,sigma2_MKT,lambda,NumAssets,w_MKT,mu_MKT)

%% Point 2.b) Building Our Views

v = 3;                  % number of views.
tau = 1/length(logret); % 1/N_obs.
Omega = zeros(v);       % uncertainty of views

[P,q] = ComputeViews(v,NumAssets,table);

for i = 1:v
    Omega(i,i) = tau * P(i,:) * V * P(i,:)';
end

% Print
PrintViews(q);

%% Point 2.c) Calculate Posterior Expected Returns

C = tau * V;
A = (C\eye(NumAssets) + P'/Omega*P);
b = (P'/Omega*q + C\mu_MKT);
mu_BL = A \ b;

% Post covariance
covBL = inv(P'/Omega*P + inv(C));

% Print
PrintExpRetBLM(NumAssets,mu_BL,mu_MKT,nm)

%% Point 2.d) Compute Efficient Frontier with Posterior Returns

% Std constraints
ub_BL = ones(NumAssets, 1); % no more 30% on single asset

% Frontier with BL parameters
ret_range_BL = linspace(min(mu_BL), max(mu_BL), 100);
FrontierVola_BL = zeros(1, length(ret_range_BL));
FrontierRet_BL = zeros(1, length(ret_range_BL));
WeightsFrontier_BL = zeros(NumAssets, length(ret_range_BL));

for i = 1:length(ret_range_BL)
    r = ret_range_BL(i);
    Aeq_temp = [ones(1, NumAssets); mu_BL'];
    beq_temp = [1; r];
    
    [w_opt, ~, exitflag] = fmincon(fun, x0, [], [], Aeq_temp, beq_temp, lb, ub_BL, [], options);
    
    if exitflag > 0
        FrontierVola_BL(i) = sqrt(w_opt' * V * w_opt);
        FrontierRet_BL(i) = w_opt' * mu_BL;
        WeightsFrontier_BL(:, i) = w_opt;
    else
        FrontierVola_BL(i) = NaN;
        FrontierRet_BL(i) = NaN;
    end
end

% Portfolio E: MV
[vol_MVP_BL, idx_MVP_BL] = min(FrontierVola_BL);
ret_MVP_BL = FrontierRet_BL(idx_MVP_BL);
w_MVP_BL = WeightsFrontier_BL(:, idx_MVP_BL);

% Portfolio F: MSR
SharpeFrontier_BL = (FrontierRet_BL - rf) ./ FrontierVola_BL;
[~, idx_MSRP_BL] = max(SharpeFrontier_BL);
w_MSRP_BL = WeightsFrontier_BL(:, idx_MSRP_BL);
vol_MSRP_BL = FrontierVola_BL(idx_MSRP_BL);
ret_MSRP_BL = FrontierRet_BL(idx_MSRP_BL);

Print_Ptfs(ret_MVP_BL, vol_MVP_BL, w_MVP_BL, 'E (BL MVP)')
Print_Ptfs(ret_MSRP_BL, vol_MSRP_BL, w_MSRP_BL, 'F (BL MSRP)')

%% Point 3.a) Compute the Portfolio with Maximum Diversification Ratio

lb = zeros(NumAssets,1);
ub = 0.25 * ones(NumAssets,1);

% Group constraints for this exercise
% Cyclical >= 20%  ->  -sum(w_cyclical) <= -0.20
A_cyc = -cyclical_assets';
b_cyc = -0.20;

% Defensive <= 50%
A_def = defensive_assets';
b_def = 0.50;

% Combine inequality constraints
Aineq_3 = [A_cyc; A_def];
bineq_3 = [b_cyc; b_def];

% Equality constraint: fully invested
Aeq_3 = ones(1, NumAssets);
beq_3 = 1;

sigma_i = sqrt(diag(V));                            % individual volatilities
fun_DR = @(w) - (w' * sigma_i) / sqrt(w' * V * w);  % maximize DR <=> minimize -DR

% Frontier grid based on volatility range
target_Vol = linspace(min(VolaPtfs), max(VolaPtfs), 100);

DR_vals = zeros(size(target_Vol));
weights_DR_frontier = zeros(NumAssets, length(target_Vol));

for i = 1:length(target_Vol)

    sigma_tar = target_Vol(i);

    % Nonlinear equality: portfolio volatility = target
    nonlcon = @(w) deal([], sqrt(w' * V * w) - sigma_tar);

    [w_opt, fval, exitflag] = fmincon(fun_DR, x0, ...
                                      Aineq_3, bineq_3, ...
                                      Aeq_3, beq_3, ...
                                      lb, ub, ...
                                      nonlcon, options);

    if exitflag > 0
        weights_DR_frontier(:,i) = w_opt;
        DR_vals(i) = -fval; % real DR
    else
        DR_vals(i) = NaN;
    end
end

[~, idx_MDR] = max(DR_vals);
w_opt_MDR = weights_DR_frontier(:, idx_MDR);

% Print and Plot
Print_MDR_Ptf(w_opt_MDR, DR_vals, target_Vol,idx_MDR)
Plot_DR_Frontier(target_Vol, DR_vals);

%% Compute the Portfolio with Maximum Entropy in risk contributions

k_fun = @(w) (w .* (V*w)) ./ (w' * V * w); % vettore delle risk contributions normalizzate
eps_  = 1e-12;                             % per evitare log(0)

fun2  = @(w) sum( k_fun(w) .* log( k_fun(w) + eps_ ) );

Entropy_vals = zeros(size(target_Vol));
weights_ENT_frontier = zeros(NumAssets, length(target_Vol));

for j = 1:length(target_Vol)

    sigma_tar = target_Vol(j);
    nonlcon   = @(w) deal([], sqrt(w' * V * w) - sigma_tar);  % vincolo: sigma(w) = sigma_tar
    
    [w_opt, fval, exitflag] = fmincon(fun2, x0, Aineq_3, bineq_3, ...
                                      Aeq_3, beq_3, lb, ub, ...
                                      nonlcon, options);

    if exitflag > 0  % soluzione trovata
        weights_ENT_frontier(:, j) = w_opt;
        Entropy_vals(j)            = -fval;   % minus perché f = sum(k log k) ≤ 0
    else 
        Entropy_vals(j) = NaN;
    end 
end

[~, idx_ME] = max(Entropy_vals);
w_opt_ME = weights_ENT_frontier(:,idx_ME); % portfolio H.

% Print and Plot
Print_ME_Ptf(w, Entropy_vals, target_Vol, idx_ME)
Plot_Entropy_Frontier(target_Vol, Entropy_vals)

%% 3.b) Compare this ptfs with the equally weighted benchmark in terms of: 
%       DR, Vol, Sharpe Ratio, Herfindahl index

w_eq = (1/NumAssets)*ones(NumAssets,1);  % equally weighted (colonna)ß

% Diversification Ratio

DR_eq  = (w_eq'      * sigma_i) / sqrt(w_eq'      * V * w_eq);
DR_MDR = (w_opt_MDR' * sigma_i) / sqrt(w_opt_MDR' * V * w_opt_MDR);
DR_ME  = (w_opt_ME'  * sigma_i) / sqrt(w_opt_ME'  * V * w_opt_ME);

% Volatility

Vol_eq  = sqrt(w_eq'      * V * w_eq);
Vol_MDR = sqrt(w_opt_MDR' * V * w_opt_MDR);
Vol_ME  = sqrt(w_opt_ME'  * V * w_opt_ME);

% Sharpe Ratio

SR_eq  = (w_eq'      * ExpRet' - rf) / Vol_eq;
SR_MDR = (w_opt_MDR' * ExpRet' - rf) / Vol_MDR;
SR_ME  = (w_opt_ME'  * ExpRet' - rf) / Vol_ME;

% Herfindahl Index

HI_eq  = sum(w_eq.^2);
HI_MDR = sum(w_opt_MDR.^2);
HI_ME  = sum(w_opt_ME.^2);

Print_Portfolio_Comparison(DR_eq, DR_MDR, DR_ME, ...
                           Vol_eq, Vol_MDR, Vol_ME, ...
                           SR_eq, SR_MDR, SR_ME, ...
                           HI_eq, HI_MDR, HI_ME)

%% 4.a) PCA Analysis

muR    = mean(logret);       % 1 x N
sigmaR = std(logret);        % 1 x N

% Standardize returns
RetStd = (logret - muR) ./ sigmaR;   % T x N

k = 5;  % number of principal components

[Loadings_All, Scores_All, latent, ~, explained] = pca(RetStd, 'NumComponents', k);

cumVar = cumsum(latent) / sum(latent);
target_var = 0.85; % Soglia 85%
k = find(cumVar >= target_var, 1, 'first');

ExplainedVar = latent(1:k) / sum(latent);

Loadings = Loadings_All(:, 1:k); 
Scores   = Scores_All(:, 1:k);

covarFactor = cov(Scores);           % k x k

% Reconstructed standardized returns and residuals
RetStd_hat   = Scores * Loadings';   % T x N
epsilon_std  = RetStd - RetStd_hat;  % T x N
Psi_std      = var(epsilon_std,0,1); % 1 x N

Lambda  = diag(sigmaR);
D       = diag(Psi_std);

CovarPCA = Lambda * (Loadings * covarFactor * Loadings' + D) * Lambda;

v1 = Loadings(:, 1);

A = v1';      % w' * v1 <= 0.5
b = 0.5;

% Reconstructed returns in original units
reconReturn    = RetStd_hat .* sigmaR + muR;
unexplainedRetn = logret - reconReturn;

% Cumulative explained variance (all components)
CumExplVar = cumsum(explained);   % "explained" è già in percentuale
n_pc       = 1:length(explained);

% Print and Plots
fprintf('Selezionati k=%d fattori. Varianza Totale Spiegata: %.2f%%\n', ...
        k, sum(ExplainedVar)*100);
Plot_PCA_Variance(ExplainedVar)
Plot_PCA_Cumulative(n_pc, CumExplVar)

%% 4.b) Maximum Sharpe Ratio & Conditional Value-at-Risk Ptfs

func_sharpe = @(x) - ((muR*x) / sqrt(x'*CovarPCA*x));
[w_sharpe, fval_sharpe] = fmincon(func_sharpe, x0, A, b, Aeq, beq, lb, ub, [], options);

Print_Sharpe_Portfolio(w, fval_sharpe, CovarPCA, v1) % CHECK THIS !!!!!!!!

alpha_tail = 0.05;
target_vol_ann = 0.10; 
target_vol_daily = target_vol_ann / sqrt(252); 

% Funzione Obiettivo: Min CVaR
func_cvar = @(w) compute_historical_cvar(w, logret, alpha_tail);

% VINCOLO 1: Target Volatility 10% (Non lineare - Uguaglianza)
% Se il solver non trova soluzione, questo è il vincolo che "stringe" troppo
nonlcon_vol = @(w) deal([], sqrt(w' * cov(logret) * w) - target_vol_daily);

% VINCOLI 2 e 3: Standard Constraints (Somma=1, Max=0.25)
NumAssets = size(logret, 2);
Aeq = ones(1, NumAssets); 
beq = 1;
ub  = ones(NumAssets, 1) * 0.25; % Max weight 25% (Standard Constraint)

[w_cvar, min_cvar_val, exitflag, output] = fmincon(func_cvar, x0, [], [], Aeq, beq, lb, ub, nonlcon_vol, options);

vol_ottenuta = sqrt(w_cvar' * cov(logret) * w_cvar) * sqrt(252);

disp('-----------------------------------------');
if exitflag > 0
    disp('Ottimizzazione riuscita!');
else
    disp('ATTENZIONE: Il solver non ha trovato una soluzione fattibile.');
    disp('Probabilmente il target del 10% di volatilità è impossibile da raggiungere.');
end
disp('Target Volatilità Richiesto: 10%');
disp(['Volatilità Raggiunta: ', num2str(vol_ottenuta*100), '%']);
disp(['CVaR (5%): ', num2str(min_cvar_val)]);
disp('-----------------------------------------');