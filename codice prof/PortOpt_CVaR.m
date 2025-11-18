clear all
close all
clc

%% Read Prices & Names
load array_prices.mat
load myPrice_dt.mat
start_dt = datetime('01/01/2021', 'InputFormat', 'dd/MM/yyyy'); 
end_dt   = datetime('01/06/2022', 'InputFormat', 'dd/MM/yyyy');

rng = timerange(start_dt, end_dt,'closed'); 
subsample = myPrice_dt(rng,:);

prices_val = subsample.Variables;
dates_ = subsample.Time;
%% Compute Moments
LogRet = tick2ret(prices_val, 'Method','Continuous');
ExpRet = mean(LogRet);
CovMatrix = cov(LogRet);

%% VaR - Historical Simulation - h = 1
% We compute VaR & ES for Equally Weighted Portfolio
ConfLevel = [0.95, 0.99];
weightsEW = ones(size(prices_val,2),1).*1/size(prices_val,2);
pRet = weightsEW'*LogRet';
VaR_95 = quantile(pRet, 1-ConfLevel(1,1));
VaR_99 = quantile(pRet, 1-ConfLevel(1,2));

% Expected Shortfall
ES_95 = mean(pRet(pRet<VaR_95));
ES_99 = mean(pRet(pRet<VaR_99));

% Plot
f = figure();
histogram(pRet)
hold on
xline(VaR_95, 'LineWidth', 4,'Color', 'r')
hold on 
xline(VaR_99, 'LineWidth', 4,'Color', 'm')
legend('Profit & Loss Distribution', 'VaR 95% confidence level', 'VaR 99% confidence level')

%% VaR - Normal distribution: h = 1
mu = mean(pRet);
std_ = std(pRet);

VaR_95Norm = mu-std_*norminv(0.95);
VaR_99Norm = mu-std_*norminv(0.99);

% Matlab Function portvrisk
ValueAtRisk95 = portvrisk(mu, std_, 1-ConfLevel(1,1));
ValueAtRisk99 = portvrisk(mu, std_, 1-ConfLevel(1,2));

% Expected Shortfall
ES_95Norm = mean(pRet(pRet<VaR_95Norm));
ES_99Norm = mean(pRet(pRet<VaR_99Norm));


% Plot
f1 = figure();
histogram(pRet)
hold on
xline(VaR_95Norm, 'LineWidth', 4,'Color', 'r')
hold on 
xline(VaR_99Norm, 'LineWidth', 4, 'Color', 'm')
legend('Profit & Loss Distribution', 'VaR 95% confidence level', 'VaR 99% confidence level')

%% VaR-based optimization: 
% Objective: maximize expected return & Constraint: portfolio VaR >= targetVaR
N = size(LogRet,2);
fun = @(x) -(ExpRet*x);  % maximize return -> minimize negative
x0 = ones(N,1)/N;
lb = zeros(1,N)+0.001;
ub = ones(1,N);
Aeq = ones(1,N);
beq = 1;
alpha = 0.95;
% Target VaR (95%) not worse than -5%
tgtVaR = -0.05;

% Nonlinear constraint function for VaR
nonlinconVaR = @(w) deal( quantile(LogRet*w,1-alpha) - tgtVaR, [] );
% Solve with fmincon (non-convex)
options = optimoptions('fmincon','Display','iter','Algorithm','sqp'); %sqp → Sequential Quadratic Programming
w_VaR = fmincon(fun, x0, [], [], Aeq, beq, lb, ub, nonlinconVaR, options);

% Evaluate VaR of the optimized portfolio
VaR_port = quantile(LogRet*w_VaR,1-alpha);
disp(['Optimized VaR (95%): ', num2str(VaR_port)]);
%% CVaR (Expected Shortfall) optimization - convex problem
% Minimize portfolio CVaR (Rockafellar & Uryasev formulation)
% Variables:
% w = portfolio weights (N x 1)
% eta = VaR threshold (1 x 1)
% u_i = slack variables for losses beyond VaR (t x 1)
% Objective: minimize [eta + 1/((1-alpha)*N)*sum(u_i)]

t = size(LogRet, 1); % number of scenarios
N = size(LogRet, 2); % number of assets

% Variables: [w(1:N), eta, ui(1:t)]
% vector of objective function coefficients, used to map the variables into the objective function and constraints
f = [zeros(N,1); 1; (1/((1-alpha)*t))*ones(t,1)]; 

% Constraints:
% u_i >= 0 and u_i >= -R*w - eta
A1 = [-LogRet, -ones(t,1), -eye(t)];   % u_i >= -R*w - eta
A2 = [zeros(t,N+1), -eye(t)];     % u_i >= 0
A = [A1; A2];
b = zeros(2*t,1);

% Equality: sum(w)=1
Aeq = [ones(1,N), 0, zeros(1,t)];
beq = 1;

% Bounds
lb = [zeros(N,1); -Inf; zeros(t,1)];
ub = [];

options = optimoptions('linprog','Display','none');
x = linprog(f, A, b, Aeq, beq, lb, ub, options);

w_CVaR = x(1:N);
eta = x(N+1);
u = x(N+2:end);
CVaR_value = eta + (1/((1-alpha)*t))*sum(u);

disp(['Optimized CVaR (95%): ', num2str(CVaR_value)]);

% Compute performance metrics
portRet_VaR = LogRet*w_VaR;
portRet_CVaR = LogRet*w_CVaR;

disp('Portfolio Weights Comparison:');
T = table(w_VaR, w_CVaR, 'VariableNames', {'VaR_opt', 'CVaR_opt'});
disp(T);
%% Equity Comparison
ret = prices_val(2:end,:)./prices_val(1:end-1,:);
% VaR 95
equityVaR95 = cumprod(ret*w_VaR);
equityVaR95 = 100.*equityVaR95/equityVaR95(1);
[annRet_VaR95, annVol_VaR95, Sharpe_VaR95, MaxDD_VaR95, Calmar_VaR95] = getPerformanceMetrics(equityVaR95);
perfTable_VaR95 = table(annRet_VaR95, annVol_VaR95, Sharpe_VaR95, MaxDD_VaR95, Calmar_VaR95, 'VariableNames',["AnnRet", "AnnVol", "Sharpe", "MaxDD","Calmar"]);


% ES 95
equityES95 = cumprod(ret*w_CVaR);
equityES95 = 100.*equityES95/equityES95(1);
[annRet_ES95, annVol_ES95, Sharpe_ES95, MaxDD_ES95, Calmar_ES95] = getPerformanceMetrics(equityES95);
perfTable_ES95 = table(annRet_ES95, annVol_ES95, Sharpe_ES95, MaxDD_ES95, Calmar_ES95, 'VariableNames',["AnnRet", "AnnVol", "Sharpe", "MaxDD","Calmar"]);

% EW
wEW = 1/15*ones(15, 1);
equity_ew = cumprod(ret*wEW);
equity_ew = 100.*equity_ew/equity_ew(1);
[annRet_ew, annVol_ew, Sharpe_ew, MaxDD_ew, Calmar_ew] = getPerformanceMetrics(equity_ew);
perfTable_ew = table(annRet_ew, annVol_ew, Sharpe_ew, MaxDD_ew, Calmar_ew, 'VariableNames',["AnnRet", "AnnVol", "Sharpe", "MaxDD","Calmar"]);

% Plot
f1 = figure();
plot(dates_(2:end,1),equityVaR95, 'LineWidth', 2)
hold on
plot(dates_(2:end,1),equityES95, 'LineWidth', 2)
hold on
plot(dates_(2:end,1),equity_ew, 'LineWidth', 2)
legend('Equity VaR', 'Equity ES', 'Equity EW')