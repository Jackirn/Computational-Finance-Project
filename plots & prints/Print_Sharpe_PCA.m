function Print_Sharpe_PCA(w, fval_sharpe, Covar,ExpRet)
% PRINT_SHARPE_PORTFOLIO  Pretty-print delle metriche del portafoglio massimizzante Sharpe Ratio

    w = w(:); % Assicurarsi che sia vettore colonna

    fprintf('\n===============================================================\n');
    fprintf('                Portfolio I - Maximizing Sharpe Ratio             \n');
    fprintf('===============================================================\n');

    fprintf('Expected Return:       %8.4f%%\n', w'*ExpRet'*100);
    fprintf('Sharpe Ratio:            %.4f\n', -fval_sharpe);
    fprintf('Volatility (CovarPCA):   %.3f%%\n', sqrt(w' * Covar * w) * 100 * sqrt(252));

    fprintf('---------------------------------------------------------------\n');
    fprintf('Top 3 Holdings:\n');
    [sorted_w, idx] = sort(w, 'descend');
    for k = 1:3
        fprintf('  Asset %d: %.2f%%\n', idx(k), sorted_w(k)*100);
    end

    fprintf('===============================================================\n\n');

end
