function Print_MDR_Ptf(w, DR_value, vol_value,idx_MDR, ExpRet)
% PRINT_MDR_PTF  Pretty-print Maximum Diversification Ratio portfolio.

    fprintf('\n===============================================================\n');
    fprintf('                Portfolio G – Maximum DR Portfolio             \n');
    fprintf('===============================================================\n');

    fprintf('Expected Return:         %8.4f%%\n', w'*ExpRet'*100);
    fprintf('Diversification Ratio:   %8.4f\n', DR_value(idx_MDR));
    fprintf('Volatility:              %8.3f%%\n', vol_value(idx_MDR) * 100);

    fprintf('---------------------------------------------------------------\n');
    fprintf('Top 3 Holdings:\n');

    [sorted_w, idx] = sort(w, 'descend');

    for k = 1:3
        fprintf('  %-12s   %8.2f%%\n', sprintf('Asset %d', idx(k)), sorted_w(k)*100);
    end

    fprintf('===============================================================\n\n');

end
