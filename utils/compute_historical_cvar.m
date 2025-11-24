function cvar = compute_historical_cvar(w, returns, alpha)
    port_returns = returns * w;
    % Definizione Perdite = -Rendimenti
    losses = -port_returns;
    % VaR è il quantile (1-alpha) delle perdite
    VaR = prctile(losses, (1-alpha)*100);
    % CVaR è la media delle perdite superiori al VaR
    cvar = mean(losses(losses >= VaR));
end