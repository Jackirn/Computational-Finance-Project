clear all
close all
clc

%% Read Prices
path_map        = 'C:\Users\ginevra.angelini\OneDrive - Anima SGR S.p.A\Desktop\lezioni_poli\lezioni\Lezione3\';
filename        = 'geo_index_prices.xlsx';

table_prices = readtable(strcat(path_map, filename));
%% Transform prices from table to timetable
dt = table_prices(:,1).Variables;
values = table_prices(:,2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm); 
%% Selection of a subset of Dates
start_dt = datetime('01/05/2021', 'InputFormat', 'dd/MM/yyyy'); 
end_dt   = datetime('01/08/2021', 'InputFormat', 'dd/MM/yyyy');

rng = timerange(start_dt, end_dt,'closed'); 
subsample = myPrice_dt(rng,:);

prices_val = subsample.Variables;
dates_ = subsample.Time;
%% Calculate returns
LogRet = tick2ret(prices_val, 'Method', 'Continuous');
ExpRet = mean(LogRet);
%% Calculate Variance-Covariance Matrix
V = cov(LogRet);

%% 1.Test on Concentration Error 
p = Portfolio('AssetList', nm);
p = setDefaultConstraints(p);
P = estimateAssetMoments(p, LogRet, 'missingdata', false);
pwgt = estimateFrontier(P, 100);
[pf_Risk, pf_Retn] = estimatePortMoments(P,pwgt);
% Plot of weights
bar(pwgt', 'Stacked')
%% 1.Test on Concentration Error: Compute efficient frontier with boundaries 
LowerBound = 0.05*ones(1,8);
UpperBound = 0.8*ones(1,8);
p = setBounds(p, LowerBound, UpperBound);
MinNumAssets = 6;
MaxNumberAssets = 8;
p = setMinMaxNumAssets(p, MinNumAssets, MaxNumberAssets);
PortBound = estimateAssetMoments(p, LogRet, 'missingdata', false);
pwgt_bound = estimateFrontier(PortBound, 100);
[pf_RiskB, pf_RetnB] = estimatePortMoments(PortBound,pwgt_bound);
% Plot Weights
h = figure;
bar(pwgt', 'Stacked')
g = figure;
bar(pwgt_bound', 'Stacked')

% Plot Frontier
plot(pf_RiskB, pf_RetnB)
hold on
plot(pf_Risk, pf_Retn)
%% 2. Test the robustness of the frontier: We add small perturbations on the return of first asset
LogRet1 = LogRet;

outlier_rows = [10, 50, 60];   
LogRet1(outlier_rows,1) = LogRet1(outlier_rows,1) + 0.03;  
LogRet1(outlier_rows,3) = LogRet1(outlier_rows,3) + 0.1;

LogRet2 = LogRet;
outlier_rows = [1, 20, 35];
LogRet2(outlier_rows,6) = LogRet2(outlier_rows,6) + 0.05;  
LogRet2(outlier_rows,2) = LogRet2(outlier_rows,2) + 0.1;

LogRet3 = LogRet;
outlier_rows = [12, 32, 45];
LogRet3(outlier_rows,3) = LogRet3(outlier_rows,6) + 0.05;  
LogRet3(outlier_rows,8) = LogRet3(outlier_rows,2) + 0.09;


%% Test the robustness of the frontier: COMPUTE EFFICIENT FRONTIERS
% Compute Classical Frontiers
p = Portfolio('AssetList', nm);
p = setDefaultConstraints(p);

Port  = estimateAssetMoments(p, LogRet, 'missingdata', false);
pwgt  = estimateFrontier(Port, 100);
[pf_risk, pf_Retn]   = estimatePortMoments(Port, pwgt);

Port1 = estimateAssetMoments(p, LogRet1, 'missingdata', false);
pwgt1 = estimateFrontier(Port1, 100);
[pf_risk1, pf_Retn1] = estimatePortMoments(Port1, pwgt1);

Port2 = estimateAssetMoments(p, LogRet2, 'missingdata', false);
pwgt2 = estimateFrontier(Port2, 100);
[pf_risk2, pf_Retn2] = estimatePortMoments(Port2, pwgt2);

Port3 = estimateAssetMoments(p, LogRet3, 'missingdata', false);
pwgt3 = estimateFrontier(Port3, 100);
[pf_risk3, pf_Retn3] = estimatePortMoments(Port3, pwgt3);  


% Robust Frontier on LogRet3
[robustSigma3, robustMu3] = robustcov(LogRet3);


% Building Robust Frontier
Port3_rob = Portfolio('AssetList', nm);
Port3_rob = setDefaultConstraints(Port3_rob);
Port3_rob = setAssetMoments(Port3_rob, robustMu3', robustSigma3);

pwgt3_rob = estimateFrontier(Port3_rob, 100);
[pf_risk3_rob, pf_Retn3_rob] = estimatePortMoments(Port3_rob, pwgt3_rob);

% Plot frontiers
figure;
plot(pf_risk, pf_Retn, 'LineWidth',2); hold on;
plot(pf_risk1, pf_Retn1, 'LineWidth',1.5);
plot(pf_risk2, pf_Retn2, 'LineWidth',1.5);
plot(pf_risk3, pf_Retn3, 'LineWidth',1.5);
plot(pf_risk3_rob, pf_Retn3_rob, '--','LineWidth',2); % robusta
xlim([0 0.05])   
ylim([0 0.01]) 
title('Portfolio Frontiers with outlier and robust frontier')
legend('Original','perturbed 1','perturbed2','perturbed3','Robust3','Location','best')
xlabel('Volatility')
ylabel('Expected Return')
grid on
%% 3. Robust Frontier : Resampling - N simulations of returns assuming they are distibuted as a normal distribution with mean = ExpRet and covariance V
% Resampled Efficient Frontier with Confidence Intervals
p = Portfolio('AssetList', nm);    % nm = cell array of asset names
p = setDefaultConstraints(p); 
N = 100;        % number of simulations
nAssets = 8;    % number of assets
nPort = 100;    % points on each frontier

RiskPtfSim = zeros(nPort, N);
RetPtfSim  = zeros(nPort, N);
Weights    = zeros(nAssets, nPort, N); % store weights for each simulation

for n = 1:N
    % Simulate asset returns from multivariate normal
    R_sim = mvnrnd(ExpRet, V);
    % Simulate covariance matrix using inverse Wishart
    Cov_sim = iwishrnd(V, nAssets);
    % Create Portfolio object with simulated moments
    P_sim = setAssetMoments(p, R_sim, Cov_sim);
    % Estimate efficient frontier
    w_sim = estimateFrontier(P_sim, nPort);
    % Store weights and corresponding portfolio moments
    Weights(:,:,n) = w_sim;
    [pf_risk, pf_ret] = estimatePortMoments(P_sim, w_sim);
    RiskPtfSim(:,n) = pf_risk;
    RetPtfSim(:,n)  = pf_ret;
end

% Compute mean frontier and confidence intervals
meanRisk = mean(RiskPtfSim,2);            % mean volatility across simulations
meanRet  = mean(RetPtfSim,2);             % mean expected return

% Compute average weights portfolio
P_avg = Portfolio('AssetList', nm);
P_avg = setDefaultConstraints(P_avg);
P_avg = setAssetMoments(P_avg, ExpRet, V);

% Compute risk and return of average weights portfolio
meanWeights = mean(Weights,3);            
[meanRiskPort, meanRetPort] = estimatePortMoments(P_avg, meanWeights);

% Plot results
figure; hold on;
% Plot all simulated frontiers in light grey
plot(RiskPtfSim, RetPtfSim, 'Color',[0.8 0.8 0.8]);
% Plot mean frontier
plot(meanRisk, meanRet, 'r','LineWidth',3);
% Plot single average-weights portfolio
xlim([-0.01, 0.1])
xlabel('Volatility');
ylabel('Expected Return');
title('Resampled Efficient Frontier with 90% Confidence Interval');
legend('Simulated Frontiers','Mean Frontier','Location','best');
grid on;

%% 4. Robust Frontier: Robust Estimators 
RobustExpRet = trimmean(LogRet, 10, 1);
RobustV      = robustcov(LogRet);

% Plot mean vs trimmed mean
figure; bar([ExpRet; RobustExpRet]');
legend('Original Exp Ret','Robust Exp Ret');
xlabel('Assets'); ylabel('Expected Return');
title('Original vs Robust Expected Returns');

% Create Portfolio objects separately
PortClassical = Portfolio('AssetList', nm); 
PortClassical = setDefaultConstraints(PortClassical);
PortClassical = setAssetMoments(PortClassical, ExpRet, V);

PortRobustMean = Portfolio('AssetList', nm); 
PortRobustMean = setDefaultConstraints(PortRobustMean); 
PortRobustMean = setAssetMoments(PortRobustMean, RobustExpRet, V);

PortRobustCov  = Portfolio('AssetList', nm); 
PortRobustCov = setDefaultConstraints(PortRobustCov); 
PortRobustCov = setAssetMoments(PortRobustCov, ExpRet, RobustV);

PortRobust = Portfolio('AssetList', nm);
PortRobust = setDefaultConstraints(PortRobust); 
[RobustV, RobustMean_new] = robustcov(LogRet);
PortRobust = setAssetMoments(PortRobust, RobustMean_new, RobustV);

% Estimate frontiers
pwgtC  = estimateFrontier(PortClassical, 100); 
[riskC,retC] = estimatePortMoments(PortClassical, pwgtC);

pwgtRM = estimateFrontier(PortRobustMean, 100); 
[riskRM,retRM] = estimatePortMoments(PortRobustMean, pwgtRM);

pwgtRC = estimateFrontier(PortRobustCov, 100);  
[riskRC,retRC] = estimatePortMoments(PortRobustCov, pwgtRC);

pwgtR  = estimateFrontier(PortRobust, 100);     
[riskR,retR] = estimatePortMoments(PortRobust, pwgtR);

% Plot
figure; hold on;
plot(riskC,retC,'k','LineWidth',2);
plot(riskRM,retRM,'r','LineWidth',2);
plot(riskRC,retRC,'b','LineWidth',2);
plot(riskR,retR,'g--','LineWidth',2);
legend('Classical','Robust Mean','Robust Cov','Robust Mean & Cov','Location','best')
xlabel('Volatility'); ylabel('Expected Return');
title('Classical vs Robust Efficient Frontiers');
grid on;


%% 5. Robust Frontier: Bayes-Stein Shrinkage Estimator for Mean
N = size(LogRet,2);       % number of assets
T = size(LogRet,1);       % number of observations

% Classical expected returns and covariance
ExpRet = mean(LogRet);    
V = cov(LogRet);

% Compute Bayes-Stein shrinkage parameter
x = ExpRet - mean(ExpRet);           % deviations from grand mean
lambda = ((N+2)*(T-1))/((T-N-2)*(x*inv(V)*x')); % scalar shrinkage intensity
alpha  = lambda/(lambda+T);

% Compute Bayes-Stein shrunk mean
BSmean = (1-alpha)*ExpRet + alpha*mean(ExpRet)*ones(1,N);

fprintf('Bayes-Stein shrinkage alpha: %.4f\n', alpha);
fprintf('Original vs Shrunk Expected Returns:\n');
disp(table(ExpRet', BSmean', 'VariableNames', {'Classical','BayesStein'}));


% Classical Frontier
PortClassical = Portfolio('AssetList', nm);
PortClassical = setDefaultConstraints(PortClassical);
PortClassical = setAssetMoments(PortClassical, ExpRet, V);

pwgtClassical = estimateFrontier(PortClassical, 100);
[pf_riskC, pf_RetnC] = estimatePortMoments(PortClassical, pwgtClassical);


% Bayes-Stein Frontier
PortBS = Portfolio('AssetList', nm);
PortBS = setDefaultConstraints(PortBS);
PortBS = setAssetMoments(PortBS, BSmean, V);

pwgtBS = estimateFrontier(PortBS, 100);
[pf_riskBS, pf_RetnBS] = estimatePortMoments(PortBS, pwgtBS);


% Plot comparison
figure; hold on;
plot(pf_riskC, pf_RetnC,'k','LineWidth',2);        % Classical
plot(pf_riskBS, pf_RetnBS,'r--','LineWidth',2);    % Bayes-Stein
xlabel('Volatility'); ylabel('Expected Return');
title('Classical vs Bayes-Stein Efficient Frontier');
legend('Classical','Bayes-Stein','Location','best');
grid on;
