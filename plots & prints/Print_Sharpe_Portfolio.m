function Print_Sharpe_Portfolio(w, fval_sharpe, CovarPCA, v1)
% PRINT_SHARPE_PORTFOLIO  Pretty-print delle metriche del portafoglio massimizzante Sharpe Ratio

    w = w(:); % Assicurarsi che sia vettore colonna

    fprintf('\n===============================================================\n');
    fprintf('                Portfolio Maximizing Sharpe Ratio             \n');
    fprintf('===============================================================\n');

    fprintf('Sharpe Ratio: %.4f\n', -fval_sharpe);
    fprintf('Volatility (CovarPCA): %.3f%%\n', sqrt(w' * CovarPCA * w) * 100);
    fprintf('Exposure to v1: %.4f\n', w' * v1);

    fprintf('---------------------------------------------------------------\n');
    fprintf('Top 3 Holdings:\n');
    [sorted_w, idx] = sort(w, 'descend');
    for k = 1:3
        fprintf('  Asset %d: %.2f%%\n', idx(k), sorted_w(k)*100);
    end

    fprintf('===============================================================\n\n');

end
