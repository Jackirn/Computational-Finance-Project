function cvar = compute_historical_cvar(weights, returns, alpha)
    portfolio_returns = returns * weights;
    var_level = quantile(portfolio_returns, alpha);
    tail_returns = portfolio_returns(portfolio_returns <= var_level);
    cvar = mean(tail_returns);
end