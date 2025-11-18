clear 
close all
clc

%% Read Prices & Names
load array_prices.mat
load myPrice_dt.mat
start_dt = datetime('01/06/2020', 'InputFormat', 'dd/MM/yyyy'); % dt_4(1)+20;
end_dt   = datetime('01/06/2021', 'InputFormat', 'dd/MM/yyyy');

rng = timerange(start_dt, end_dt,'closed'); %fai vedere openLeft ecc 'openLeft'
subsample = myPrice_dt(rng,:);

prices_val = subsample.Variables;
dates_ = subsample.Time;
%% Compute Moments
LogRet = tick2ret(prices_val, 'Method','Continuous');
muR = mean(LogRet);
sigmaR = std(LogRet);
CovSample = cov(LogRet);
n_assets = size(LogRet,2);
%% PCA Factor Model (Statistical Multi-Factor)
% Standardize returns
RetStd = (LogRet - muR) ./ sigmaR;
%% PCA decomposition
k = 5;  % number of principal components
[factorLoading, factorRetn, latent, ~, explained] = pca(RetStd, 'NumComponents', k);
covarFactor = cov(factorRetn);

% Rescale back to original return units
Lambda = diag(sigmaR);
D_std = diag(var(LogRet - (factorRetn*factorLoading' .* sigmaR + muR))); % diagonal matrix that contains std of original assets returns
CovarPCA = Lambda * (factorLoading * covarFactor * factorLoading' + D_std) * Lambda;
reconReturn = factorRetn * factorLoading' .* sigmaR + muR;
unexplainedRetn = LogRet - reconReturn; % epsilon

% Explained variance
ExplainedVar = latent(1:k) / sum(latent);
figure;
bar(ExplainedVar*100);
title('Variance explained by each Principal Component');
xlabel('Principal Component');
ylabel('Explained Variance (%)');



CumExplVar = cumsum(explained);
n_list = linspace(1, 15, n_assets);
% plot 2
f = figure();
title('Total Percentage of Explained Variances for the first n-components')
plot(n_list,CumExplVar, 'm')
hold on
scatter(n_list,CumExplVar,'m', 'filled')
grid on
xlabel('Total number of Principal Components')
ylabel('Percentage of Explained Variances')

%% Optimization max(ret-variance)
x0 = ones(n_assets,1)/n_assets;
lb = zeros(1,n_assets);
ub = ones(1,n_assets);
Aeq = ones(1,n_assets);
beq = 1;

%% Mean-Variance Optimization (PCA model)
lambda = 5; % risk aversion coefficient
func_mv = @(x) - (muR*x - 0.5*lambda*(x'*CovarPCA*x));
[w_mv, fval_mv] = fmincon(func_mv, x0, [],[],Aeq,beq,lb,ub);
%% Maximum Sharpe Ratio Optimization (PCA model)
func_sharpe = @(x) - ((muR*x) / sqrt(x'*CovarPCA*x));
[w_sharpe, fval_sharpe] = fmincon(func_sharpe, x0, [],[],Aeq,beq,lb,ub);

%% Equity Curve
start_dt = datetime('02/06/2021', 'InputFormat', 'dd/MM/yyyy'); 
end_dt   = datetime('31/12/2021', 'InputFormat', 'dd/MM/yyyy');
rng = timerange(start_dt, end_dt,'closed');
subsample = myPrice_dt(rng,:);

prices_val = subsample.Variables;
dates_ = subsample.Time;
ret = prices_val(2:end,:) ./ prices_val(1:end-1,:) - 1;

equity_mv = cumprod(1 + ret*w_mv);
equity_sharpe = cumprod(1 + ret*w_sharpe);
equity_ew = cumprod(1 + ret*(ones(n_assets,1)/n_assets));

equity_mv = 100*equity_mv/equity_mv(1);
equity_sharpe = 100*equity_sharpe/equity_sharpe(1);
equity_ew = 100*equity_ew/equity_ew(1);

%% Plot and Compare Equity Curves
figure;
plot(dates_(2:end), equity_mv, 'LineWidth',1.4);
hold on;
plot(dates_(2:end), equity_sharpe, 'LineWidth',1.4);
plot(dates_(2:end), equity_ew, '--k','LineWidth',1);
legend('PCA Mean-Variance','PCA Max Sharpe','Equal Weight','Location','best');
xlabel('Date');
ylabel('Equity Curve (base = 100)');
grid on;

%% Portfolio Statistics
[annRet_mv, annVol_mv, Sharpe_mv, MaxDD_mv, Calmar_mv] = getPerformanceMetrics(equity_mv);
[annRet_sh, annVol_sh, Sharpe_sh, MaxDD_sh, Calmar_sh] = getPerformanceMetrics(equity_sharpe);
[annRet_ew, annVol_ew, Sharpe_ew, MaxDD_ew, Calmar_ew] = getPerformanceMetrics(equity_ew);

perfTable = table( ...
    [annRet_mv; annRet_sh;  annRet_ew], ...
    [annVol_mv; annVol_sh;  annVol_ew], ...
    [Sharpe_mv; Sharpe_sh;  Sharpe_ew], ...
    [MaxDD_mv; MaxDD_sh;  MaxDD_ew], ...
    [Calmar_mv; Calmar_sh;  Calmar_ew], ...
    'VariableNames', ["AnnRet","AnnVol","Sharpe","MaxDD","Calmar"], ...
    'RowNames', {'PCA Mean-Var','PCA MaxSharpe','Equal Weight'});

disp('=== Portfolio Performance Comparison ===');
disp(perfTable);
%% Sensitivity Analysis + Optimization MaxSharpe
k_list = [2,5,7,10,12];
x0 = rand(size(LogRet,2),1);
x0 = x0./sum(x0);
lb = zeros(1,size(LogRet,2));
ub = ones(1,size(LogRet,2));
Aeq = ones(1,size(LogRet,2));
beq = 1;
weights_pca = zeros(size(LogRet,2), size(k_list,2));
equity_matrix = zeros(size(equity_ew,1), size(k_list,2));
ExplVar = ones(size(k_list,2),1);
perf_table =  table('Size',[5 5],'VariableTypes',{'double', 'double','double','double','double'}, 'VariableNames',{'AnnRet', 'AnnVol', 'Sharpe', 'MaxDD','Calmar'},'RowNames',{'k = 2','k = 5','k = 7', 'k = 10','k = 12'});

for i= 1:size(k_list,2)
    k = k_list(i);
    [factorLoading,factorRetn,latent,~, explained] = pca(RetStd, 'NumComponents', k);
    cumulativeExplained = cumsum(explained);
    ExplVar(i,:) = cumulativeExplained(k);
    covarFactor = cov(factorRetn);
    % Rescale back to original return units
    Lambda = diag(sigmaR);
    D_std = diag(var(LogRet - (factorRetn*factorLoading' .* sigmaR + muR)));
    CovarPCA = Lambda * (factorLoading * covarFactor * factorLoading' + D_std) * Lambda;
    reconReturn = factorRetn * factorLoading' .* sigmaR + muR;
    
    func_sharpe = @(x) - ((muR*x) / sqrt(x'*CovarPCA*x));
    [w_opt, fval] = fmincon(func_sharpe, x0, [],[],Aeq, beq, lb, ub);
    weights_pca(:,i) = w_opt;
    equity_temp = cumprod(1+(ret*w_opt));
    equity_matrix(:,i)= 100.*equity_temp/equity_temp(1);
    [annRet_, annVol_, Sharpe_, MaxDD_, Calmar_] = getPerformanceMetrics(equity_temp);
    perf_table(i,:) = table(annRet_, annVol_, Sharpe_, MaxDD_, Calmar_);
end

%plot Expl Var
f1 = figure();
plot(k_list, ExplVar)
hold on 
scatter(k_list, ExplVar, 'filled')

% Plot Equity
f2 = figure();
plot(dates_(2:end,1), equity_matrix)
hold on 
plot(dates_(2:end,1),equity_ew, 'k')
grid on
legend('k = 2','k = 5', 'k = 7', 'k = 10','k =12', 'EW')