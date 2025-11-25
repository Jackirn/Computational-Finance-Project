function Print_Portfolio_J(min_vol_j, target_vol_ann, exitflag, output, ...
                           vol_ottenuta, portfolio_cvar, returns_portfolio, ...
                           w_cvar)

    fprintf('\n===============================================================\n');
    fprintf('                Portfolio J - Minimum CVaR (Target 10%% Vol)      \n');
    fprintf('===============================================================\n');

    % Sezione: Check della volatilità minima
    fprintf('Minimum Achievable Volatility (constraints J):       %6.2f%%\n', min_vol_j*100);

    if min_vol_j > target_vol_ann
        fprintf('WARNING: Target 10%% NOT reachable. Minimum possible: %6.2f%%\n', min_vol_j*100);
        target_vol_ann_adj = min_vol_j;
        fprintf('Using adjusted volatility target:                    %6.2f%%\n', target_vol_ann_adj*100);
    else
        target_vol_ann_adj = target_vol_ann;
    end

    fprintf('---------------------------------------------------------------\n');

    % Stato dell’ottimizzazione
    if exitflag > 0
        fprintf('Optimization Status:    SUCCESS\n');
    else
        fprintf('Optimization Status:    WARNING\n');
        fprintf('  Exit flag:            %d\n', exitflag);
        fprintf('  Message:              %s\n', output.message);
    end

    fprintf('---------------------------------------------------------------\n');

    % Metriche principali del portafoglio
    fprintf('Target Volatility:       %6.2f%%\n', target_vol_ann_adj*100);
    fprintf('Obtained Volatility:     %6.2f%%\n', vol_ottenuta*100);
    fprintf('CVaR (5%%):             %10.4f%%\n', portfolio_cvar*100);
    fprintf('Annual Mean Return:     %6.2f%%\n', mean(returns_portfolio)*252*100);

    sharpe = mean(returns_portfolio) / std(returns_portfolio) * sqrt(252);
    fprintf('Sharpe Ratio:            %6.3f\n', sharpe);

    fprintf('---------------------------------------------------------------\n');

    % Top 3 holdings
    fprintf('Top 3 Holdings:\n');
    [sorted_w, idx] = sort(w_cvar, 'descend');
    for k = 1:3
        fprintf('  Asset %2d: %6.2f%%\n', idx(k), sorted_w(k)*100);
    end

    fprintf('===============================================================\n\n');

end
