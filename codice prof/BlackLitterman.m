clear all
close all
clc

%% Read Prices
load myPrice_dt
load array_prices

%% Calculate returns and Covariance Matrix
Ret = tick2ret(prices_val);
numAssets = size(Ret, 2);
CovMatrix = cov(Ret);
%% Building the views
v = 3; 
% Tau is a scaling factor controlling how much we trust the prior (equilibrium) returns. 
% Smaller tau → more reliance on prior, larger tau → more weight to views
tau = 1/length(Ret);      
P = zeros(v, numAssets);  % pick matrix
q = zeros(v, 1);          % expected returns from views
Omega = zeros(v);         % uncertainty of views
% View 1: 5% annual return of Apple
assetNames = string(assetNames);

P(1, assetNames == "AAPLUSEquity") = 1;
q(1) = 0.05;
% View 2: 3% annual return of Amazon
P(2, assetNames == 'AMZNUSEquity') =1;
q(2) = 0.03;
% View 3 Google is going to outperform JPMorgan by 5% 
P(3, assetNames == 'GOOGLUSEquity') = 1;
P(3, assetNames == 'JPMUSEquity') = -1;
q(3) = 0.05;

% Compute Omega as tau*P*Cov*P' (diagonal approximation)
% We assume views are independent here, which is why Omega is diagonal. 
% Correlated views would require a full covariance matrix
% Smaller variance → higher confidence → more influence on posterior returns
for i = 1:v
    Omega(i,i) = tau * P(i,:) * CovMatrix * P(i,:)';
end

% Experiment on confidence (Omega)
% Increase confidence in Apple (smaller variance)
Omega(1,1) = 0.5 * Omega(1,1);
% Decrease confidence in Amazon (larger variance)
Omega(2,2) = 2 * Omega(2,2);

% from annual view to daily view -> We convert annual views to daily to 
% match the frequency of returns in the dataset
daysPerYear = 252;
q = q / daysPerYear;
Omega = Omega / daysPerYear;

% Plot views distribution
X_views = mvnrnd(q, Omega, 200);
figure;
hold on
for i = 1:v
    histogram(X_views(:,i), 'DisplayName', ['View ' num2str(i)], 'FaceAlpha',0.5);
end
legend
title('Distribution of Views')
hold off
%% market implied ret
% the equilibrium returns are likely equal to the implied returns from the 
% equilibrium portfolio holding. 
% In practice, the applicable equilibrium portfolio holding can be any 
% optimal portfolio that the investment analyst would use 
% in the absence of additional views on the market, such as the portfolio 
% benchmark, an index, or even the current portfolio
load cap
wMKT = cap(1:numAssets) / sum(cap(1:numAssets));  % market weights
lambda = 1.2;                                     % risk aversion coefficient
mu_mkt = lambda * CovMatrix * wMKT;               % equilibrium returns
C = tau * CovMatrix;                              % scaled prior covariance

% Plot prior distribution
X_prior = mvnrnd(mu_mkt, C, 200);
figure;
histogram(X_prior);
title('Prior Distribution of Returns (Equilibrium)');
%% Black Litterman
% muBL is the posterior expected return vector: it combines market 
% equilibrium (mu_mkt) with our views 
% weighted by their uncertainty (Omega).covBL is the posterior covariance, 
% accounting for the uncertainty in our views
muBL = (C\eye(numAssets) + P'/Omega*P) \ (P'/Omega*q + C\mu_mkt);
covBL = inv(P'/Omega*P + inv(C));

% Compare prior vs BL
TBL = table(assetNames', mu_mkt*daysPerYear, muBL*daysPerYear, ...
    'VariableNames', ["Asset","PriorReturnAnnual","BLReturnAnnual"]);
disp(TBL)% Plot Distribution
% NB: Even a view on a single asset affects other assets through the covariance matrix: 
% the Black–Litterman model propagates the impact according to cross-asset correlations. 
% Positively correlated assets see their expected returns increase, 
% while negatively correlated ones decrease

%% Analysis of contribution of each view
% We decompose the impact of each view on the final BL expected returns. 
% This helps understand which view drives which asset’s return
contrib = zeros(numAssets, v);
for i = 1:v
    P_i = P(i,:)';
    Omega_i = Omega(i,i);
    contrib(:,i) = CovMatrix * P_i / (P_i' * CovMatrix * P_i + Omega_i) * (q(i) - P_i' * mu_mkt);
end
% each contrib(:,i) shows how the expected return of each asset is adjusted because of view i

% Total impact on expected returns
muBL_contrib = mu_mkt + sum(contrib,2);

% Plot contributions
figure;
bar(contrib * daysPerYear); %Each color (bar) corresponds to a different view
xlabel('Asset Index'); %Represents the assets
ylabel('Annualized Contribution'); % shows the annualized contribution (how much the expected return is shifted by that view)
title('Contribution of Each View to BL Expected Returns');
legend("View1","View2","View3");
% A positive bar → the view increases that asset’s expected return
% A negative bar → the view decreases it
%% Portfolio Optimization
% We optimize classical MV portfolio based on sample mean returns. 
% For BL, we update expected returns and covariance with posterior values 
% and re-optimize
port = Portfolio('NumAssets', numAssets, 'Name', 'Mean-Variance');
port = setDefaultConstraints(port);
port = setAssetMoments(port, mean(Ret), CovMatrix);
pwgt = estimateFrontier(port, 100);
[pf_vola, pf_ret] = estimatePortMoments(port, pwgt);

% MaxSharpe
rf = 0;  % risk-free rate
SharpeArray = zeros(1,100);
for i = 1:100
    SharpeArray(i) = (pf_ret(i) - rf)/pf_vola(i);
end
[MaxSharpe, idxMax] = max(SharpeArray);
w_maxsharpe = pwgt(:, idxMax);


%wts = estimateMaxSharpeRatio(port);
%sum(wts-w_maxsharpe)

% Plot Sharpe vs vola
plot(pf_vola, SharpeArray, '-o', 'LineWidth', 3)
hold on
scatter (pf_vola(idxMax), MaxSharpe, 'LineWidth', 3)
%% Black-Litterman PTF
portBL = Portfolio('NumAssets', numAssets, 'Name', 'MV with BL');
portBL = setDefaultConstraints(portBL);
portBL = setAssetMoments(portBL, muBL, CovMatrix+covBL);
wtsBL = estimateMaxSharpeRatio(portBL);
[risk, ret] = estimatePortMoments(portBL, wtsBL);

% Compare classical vs BL portfolio weights
Tweights = table(assetNames', w_maxsharpe, wtsBL, ...
    'VariableNames', ["Asset","ClassicalMV","BLMV"]);
disp(Tweights)

% Plot
figure;
subplot(1,2,1)
idx = w_maxsharpe > 0.001;
pie(w_maxsharpe(idx), assetNames(idx))
title('Classical MV Portfolio')

subplot(1,2,2)
idxBL = wtsBL > 0.001;
pie(wtsBL(idxBL), assetNames(idxBL))
title('Black-Litterman MV Portfolio')
%% Impact of views on portfolio (Delta weights)
delta_weights = wtsBL - w_maxsharpe;
figure;
bar(delta_weights);
xlabel('Asset Index'); ylabel('Change in Weight');
title('Impact of Views on Portfolio Allocation');
