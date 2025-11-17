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

figure;
scatter(VolaPtfs, RetPtfs, [], SharpePtfs, 'filled');
hold on;

%% Point 1.a) Compute the Efficient Frontier

fun = @(x) x'*V*x;                     % objective = variance
ret_range = linspace(min(RetPtfs), max(RetPtfs),100);
x0 = ones(NumAssets,1)/NumAssets;      % initial guess = equal weights
lb = zeros(1,NumAssets);               % lower bound --> every w(i) must be >= 0 --> no short selling.
ub = ones(1,NumAssets);                % upper bound --> every w(i) must be <= 1.

FrontierVola = zeros(1,length(ret_range)); % we have to compute the exp_ret and volat for every point in the ret_range.
FrontierRet  = zeros(1,length(ret_range));

A_max = eye(NumAssets); % constraint: maximum exposition of every asset set to 0.3.
b_max = 0.3*ones(NumAssets,1);

% Leggi la classificazione dal mapping table
defensive_assets = contains(table.MacroGroup, 'Defensive');
neutral_assets = contains(table.MacroGroup, 'Neutral'); 
cyclical_assets = contains(table.MacroGroup, 'Cyclical');

% Neutral >= 20%: sum(w_neutral) >= 0.20
A_neutral = -neutral_assets';  % -sum(w_neutral) <= -0.20
b_neutral = -0.20;

% Defensive <= 45%: sum(w_defensive) <= 0.45  
A_defensive = defensive_assets';
b_defensive = 0.45;

A_ineq = [A_neutral; A_defensive; A_max];
b_ineq = [b_neutral; b_defensive; b_max];

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');

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

fun = @(w) w'*V*w;
x0  = ones(NumAssets,1)/NumAssets;
lb  = zeros(NumAssets,1);
ub  = ones(NumAssets,1);

Aeq_MVP = ones(1,NumAssets);
beq_MVP = 1;

%w_MVP = fmincon(fun, x0, A_ineq, b_ineq, Aeq_MVP, beq_MVP, lb, ub);
% w_MVP represents the Minimum Variance Portfolio with the constraints said above.

% MVP with the above constraints

[~, idx_MVP] = min(FrontierVola);
w_MVP = WeightsFrontier(:, idx_MVP);
vol_MVP = FrontierVola(idx_MVP);
ret_MVP = FrontierRet(idx_MVP);

Print_Ptfs(ret_MVP, vol_MVP, w_MVP, 'A (MVP)')

% MSRP with the above constraints

rf = 0;
SharpeFrontier = (FrontierRet - rf) ./ FrontierVola;
[~, idx_MSRP] = max(SharpeFrontier);
w_MSRP = WeightsFrontier(:, idx_MSRP);

vol_MSRP = FrontierVola(idx_MSRP);
ret_MSRP = FrontierRet(idx_MSRP);

Print_Ptfs(ret_MSRP, vol_MSRP, w_MSRP, 'B (MSRP)')

% Plot
Plot_Frontier(FrontierVola,FrontierRet,NumAssets,V,ExpRet,SharpeFrontier)

%% Point 1.b) Robust Frontier - Resampling Approach

% Parameters for resampling
M = 500;  % number of simulations (puoi aumentare a 500 per risultati più stabili)
T = size(logret, 1);  % number of daily observations in-sample
nPort = 150;  % number of points on frontier

% Estimate initial parameters from DAILY data
e0_daily = mean(logret)';      % Daily expected returns (column vector)
V0_daily = cov(logret);        % Daily covariance matrix

RiskPtfSim = zeros(nPort, M);
RetPtfSim = zeros(nPort, M);
WeightsSim = zeros(NumAssets, nPort, M);

min_ret = min(ExpRet);
max_ret = max(ExpRet);
Ret_range = linspace(min_ret * 0.9, max_ret * 1.1, nPort);

% Optimization setup (same constraints as point a)
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
                      'MaxFunctionEvaluations', 10000, 'MaxIterations', 1000);
fprintf('Total simulations: %d\n', M);
fprintf('Points per frontier: %d\n', nPort);
fprintf('Total optimizations: %d\n\n', M * nPort);

valid_simulations = 0;
% Main resampling loop
for m = 1:M 
    if mod(m, 50) == 0
        fprintf('Completed %d/%d simulations...\n', m, M);
    end
    % Generate sample of T daily observations
    % Simulate T days of returns for all assets from multivariate normal
    Cov_sim_daily = iwishrnd(V0_daily, NumAssets);        % may we can add +10 to NumAssets to make it more stable.
    e_sim_daily = mvnrnd(e0_daily, Cov_sim_daily/NumAssets)';  % Rendimenti con incertezza

    % Estimate parameters for this sample
    e_i = e_sim_daily * 252;             % Annualized returns
    V_i = Cov_sim_daily * 252;           % Annualized covariance
    
    % Create portfolio with simulated moments
    frontier_valid = true;
    
    for j = 1:nPort
        targetRet = Ret_range(j);
        
        % Define constraints (stessi del punto 1.a)
        Aeq_temp = [ones(1, NumAssets); e_i'];
        b_eq_temp = [1; targetRet];
        
        fun_temp = @(x) x' * V_i * x;
        
        [w_opt, ~, exitflag] = fmincon(fun_temp, x0, A_ineq, b_ineq, Aeq_temp, b_eq_temp, ...
                                      lb, ub, [], options);
        
        if exitflag > 0
            WeightsSim(:, j, m) = w_opt;
            RiskPtfSim(j, m) = sqrt(w_opt' * V_i * w_opt);
            RetPtfSim(j, m) = w_opt' * e_i;
        else
            frontier_valid = false;
            break;
        end
    end

    if frontier_valid
        valid_simulations = valid_simulations + 1;
    else
        WeightsSim(:, :, m) = NaN;
        RiskPtfSim(:, m) = NaN;
        RetPtfSim(:, m) = NaN;
    end
end

valid_mask = ~isnan(squeeze(WeightsSim(1, 1, :)));
WeightsSim_valid = WeightsSim(:, :, valid_mask);
RiskPtfSim_valid = RiskPtfSim(:, valid_mask);
RetPtfSim_valid = RetPtfSim(:, valid_mask);

% Calculate final robust frontier as AVERAGE of all frontiers
meanWeights = mean(WeightsSim, 3, 'omitnan');

% Initialize vol and ret
RobustRisk = zeros(nPort, 1);
RobustRet = zeros(nPort, 1);

for j = 1:nPort
    RobustRisk(j) = sqrt(meanWeights(:, j)' * V * meanWeights(:, j));
    RobustRet(j) = meanWeights(:, j)' * ExpRet';
end

% Take just valid points
valid_points = ~isnan(RobustRisk) & ~isnan(RobustRet);
RobustRisk = RobustRisk(valid_points);
RobustRet = RobustRet(valid_points);
meanWeights = meanWeights(:, valid_points);

% RMVP 
[vol_MVP_RF, idx_MVP_RF] = min(RobustRisk);
ret_MVP_RF = RobustRet(idx_MVP_RF);
w_MVP_RF = meanWeights(:, idx_MVP_RF);

Print_Ptfs(ret_MVP_RF, vol_MVP_RF, w_MVP_RF, 'C (Robust MVP)')

% RMSRP
SharpeRatios_RF = (RobustRet - rf) ./ RobustRisk;
[~, idx_MSRP_RF] = max(SharpeRatios_RF);
w_MSRP_RF = meanWeights(:, idx_MSRP_RF);
vol_MSRP_RF = RobustRisk(idx_MSRP_RF);
ret_MSRP_RF = RobustRet(idx_MSRP_RF);

%check_constraints(w_MVP_RF, table, 'Portfolio C (Robust MVP)');
%check_constraints(w_MSRP_RF, table, 'Portfolio D (Robust MSRP)');

Print_Ptfs(ret_MSRP_RF, vol_MSRP_RF, w_MSRP_RF, 'D (Robust MSRP)')

Plot_Robust_Frontier(RobustRisk, RobustRet, NumAssets, V, ExpRet, w_MVP_RF, w_MSRP_RF, rf)
Plot_Both_Frontiers(FrontierVola, FrontierRet, RobustRisk, RobustRet, ...
                              w_MVP, w_MSRP, w_MVP_RF, w_MSRP_RF, V, ExpRet, rf)

%% Point 2.a) BLM equilibrium returns

cap_weights = table_capw2{:, 3};
w_MKT = cap_weights(1:NumAssets) ./ sum(cap_weights(1:NumAssets)); % market weights

% Calculate market portfolio statistics
ExpRet_MKT = w_MKT' * ExpRet'; % Expected return of market portfolio
sigma2_MKT = w_MKT' * V * w_MKT; % Variance of market portfolio

lambda = (ExpRet_MKT - rf) / sigma2_MKT; % Common approach for computing lambda

mu_MKT = lambda * V * w_MKT; % Implied Equilibrium Return Vector

%fprintf('\nEquilibrium Returns:\n');
%for i = 1:length(mu_MKT)
%    fprintf('  %s: %.4f\n', nm{i}, mu_MKT(i));
%end

%% Point 2.b) Building Our Views

v = 3; % number of views.
tau = 1/length(logret); % 1/N_obs.

P = zeros(v, NumAssets);  
q = zeros(v, 1);          % expected returns from views
Omega = zeros(v);         % uncertainty of views

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

% compute Omega, assuming that views are independent --> reason why Omega% is diagonal.

for i = 1:v
    Omega(i,i) = tau * P(i,:) * V * P(i,:)';
end

fprintf('\nBlack-Litterman Views Verification:\n');
fprintf('View 1: Cyclical > Neutral by %.1f%%\n', q(1)*100);
fprintf('View 2: Asset_10 < Defensive avg by %.1f%%\n', abs(q(2))*100);
fprintf('View 3: Asset_2 > Asset_13 by %.1f%%\n', q(3)*100);

%% Point 2.c) Calculate Posterior Expected Returns

C = tau * V;
A = (C\eye(NumAssets) + P'/Omega*P);
b = (P'/Omega*q + C\mu_MKT);

% Post exp ret
mu_BL = A \ b;

% Post covariance
covBL = inv(P'/Omega*P + inv(C));

fprintf('\nBlack-Litterman Posterior Returns - Top Changes:\n');
fprintf('Asset\t\tPrior\t\tPosterior\tChange\t\t%% Change\n');
changes = mu_BL - mu_MKT;
[~, sorted_idx] = sort(abs(changes), 'descend');

for i = 1:min(8, NumAssets)
    idx = sorted_idx(i);
    pct_change = (mu_BL(idx) - mu_MKT(idx)) / abs(mu_MKT(idx)) * 100;
    fprintf('%s\t\t%.4f\t\t%.4f\t\t%+.4f\t\t%+.1f%%\n', ...
            nm{idx}, mu_MKT(idx), mu_BL(idx), changes(idx), pct_change);
end

view_impact = norm(mu_BL - mu_MKT);
fprintf('\nTotal impact from views: %.6f (%.2f%% of average prior return)\n', ...
        view_impact, view_impact/mean(abs(mu_MKT))*100);

%% Point 2.d) Compute Efficient Frontier with Posterior Returns

% Std constraints
Aeq_BL = ones(1, NumAssets);
beq_BL = 1;
lb_BL = zeros(NumAssets, 1);
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
    
    [w_opt, ~, exitflag] = fmincon(fun, x0, [], [], Aeq_temp, beq_temp, lb_BL, ub_BL, [], options);
    
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

sigma_i = sqrt(diag(V)); % equal to use std(logret)

target_Vol = linspace(min(VolaPtfs),max(VolaPtfs),100); 

DR_vals = zeros(size(target_Vol));
weights_DR_frontier = zeros(NumAssets, length(target_Vol));

fun = @(w) - (w'*sigma_i) / sqrt(w'*V*w); % maximize DR <--> minimize -DR
Aeq = ones(1,NumAssets); 
beq = 1; % sum of w(i) equal to 1
lb = zeros(NumAssets,1); % w(i) >= 0
ub = 0.25*ones(NumAssets,1); % w(i) <= 0.25
w0 = ones(NumAssets,1)/NumAssets; % initial guess: equally weighted ptf

% Cyclical >= 20%: sum(w_cyclical) >= 0.20
A_cyclical = -cyclical_assets';
b_cyclical = -0.20;

% Defensive <= 50%: sum(w_defensive) <= 0.50  
A_defensive = defensive_assets';
b_defensive = 0.50;

Aineq = [A_cyclical; A_defensive];
bineq = [b_cyclical; b_defensive];

for i = 1:length(target_Vol)

    sigma_tar = target_Vol(i);
    nonlcon = @(w) deal([], sqrt(w'*V*w) - sigma_tar); % we now have a non linear equality constraint
    
    [w_opt, fval, exitflag] = fmincon(fun, w0, Aineq, bineq, Aeq, beq, lb, ub, nonlcon);

    if exitflag > 0 % soluzione trovata
        weights_DR_frontier(:,i) = w_opt;
        DR_vals(i) = -fval; %(w_opt'*sigma_i)/sqrt(w_opt'*V*w_opt)

    else 
        DR_vals(i) = NaN;

    end 

end

[~, idx_MDR] = max(DR_vals);
w_opt_MDR = weights_DR_frontier(:,idx_MDR); % portfolio G.

figure;
plot(target_Vol, DR_vals, 'LineWidth', 2)
xlabel('Portfolio Volatility')
ylabel('Diversification Ratio')
title('Diversification–Risk Frontier')
grid on

%% Compute the Portfolio with Maximum Entropy in risk contributions

fun2 = @(w) sum({[w.*(V*w)]/(w'*V*w)}.*log([w.*(V*w)]/(w'*V*w)));

Entropy_vals = zeros(size(targetVol));
weights_ENT_frontier = zeros(NumAssets, length(targetVol));

for k = 1:length(target_Vol)

    sigma_tar = target_Vol(i);
    nonlcon = @(w) deal([], sqrt(w'*V*w) - sigma_tar); % we now have a non linear equality constraint

    [w_opt, fval, exitflag] = fmincon(fun2, x0, Aineq,bineq,Aeq, beq, lb, ub, nonlcon);

    if exitflag > 0 % soluzione trovata
        weights_ENT_frontier(:,k) = w_opt;
        Entropy_vals(k) = -fval;

    else 
        Entropy_vals(k) = NaN;

    end 

end

[~, idx_ME] = max(Entropy_vals);
w_opt_ME = weights_ENT_frontier(:,idx_ME); % portfolio H.

figure;
plot(target_Vol, Entropy_vals, 'LineWidth', 2)
xlabel('Portfolio Volatility')
ylabel('Entropy in risk contributions')
title('Entropy–Risk Frontier')
grid on




