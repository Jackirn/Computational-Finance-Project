clear all 
close all
clc

%% From table to timetable

baseDir = fileparts(mfilename('fullpath')); 
csv     = fullfile(baseDir, 'csv');         
addpath(csv, '');                           
path_map = [csv filesep];
filename = 'asset_prices.csv';

table_prices = readtable(strcat(path_map, filename));

dt = table_prices(:,1).Variables;
values = table_prices(:,2:end).Variables;
nm = table_prices.Properties.VariableNames(2:end);

myPrice_dt = array2timetable(values, 'RowTimes', dt, 'VariableNames', nm);

%% Selection of a subset of Dates (In-Sample Dates)

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

%% Compute the Efficient Frontier

fun = @(x) x'*V*x;                      % objective = variance
ret_range = linspace(min(RetPtfs), max(RetPtfs),100);
x0 = ones(NumAssets,1)/NumAssets;      % initial guess = equal weights
lb = zeros(1,NumAssets);               % lower bound --> every w(i) must be >= 0 --> no short selling.
ub = ones(1,NumAssets);                % upper bound --> every w(i) must be <= 1.

FrontierVola = zeros(1,length(ret_range)); % we have to compute the exp_ret and volat for every point in the ret_range.
FrontierRet  = zeros(1,length(ret_range));

A_max = eye(NumAssets); % constraint: maximum exposition of every asset set to 0.3.
b_max = 0.3*ones(NumAssets,1);

A1 = [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0]; % constraint: group Neutral exposition >= 20%
b1 = -0.20;

A2 = [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]; % constraint: group Defensive exposition <= 45%
b2 = 0.45;

A_ineq = [A1; A2; A_max];
b_ineq = [b1; b2; b_max];

for i = 1:length(ret_range)
    r = ret_range(i);
    Aeq = [ones(1,NumAssets); ExpRet]; % added constraints: sum weights =1, target return
    beq = [1; r];
    w_opt = fmincon(fun, x0, A_ineq, b_ineq, Aeq, beq, lb, ub); % fmincon is a nonlinear programming solver.
    FrontierVola(i) = sqrt(w_opt'*V*w_opt);
    FrontierRet(i)  = w_opt'*ExpRet';
end

fun = @(w) w'*V*w;
x0  = ones(NumAssets,1)/NumAssets;
lb  = zeros(NumAssets,1);
ub  = ones(NumAssets,1);

Aeq_MVP = ones(1,NumAssets);
beq_MVP = 1;

w_MVP = fmincon(fun, x0, A_ineq, b_ineq, Aeq_MVP, beq_MVP, lb, ub);
% w_MVP represents the Minimum Variance Portfolio with the constraints said above.

WeightsEW = (1/NumAssets).*ones(NumAssets,1);

plot(FrontierVola, FrontierRet,'LineWidth', 4)
hold on
scatter(sqrt(WeightsEW'*V*WeightsEW), WeightsEW'*ExpRet', 'filled')
hold on
scatter(sqrt(w_MVP'*V*w_MVP), w_MVP'*ExpRet','filled')
legend('Efficient Frontier','EW', 'MVP')