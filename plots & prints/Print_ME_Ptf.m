function Print_ME_Ptf(w, Entropy_vals, vol_value, idx_ME)
% PRINT_ME_PTF  Pretty-print Maximum Entropy portfolio.

    fprintf('\n===============================================================\n');
    fprintf('                Portfolio H – Maximum Entropy Portfolio       \n');
    fprintf('===============================================================\n');

    fprintf('Entropy:    %8.4f\n', Entropy_vals(idx_ME));
    fprintf('Volatility: %8.3f%%\n', vol_value(idx_ME) * 100);

    fprintf('---------------------------------------------------------------\n');
    fprintf('Top 3 Holdings:\n');

    [sorted_w, idx] = sort(w, 'descend');
    for k = 1:3
        fprintf('  %-12s   %8.2f%%\n', sprintf('Asset %d', idx(k)), sorted_w(k)*100);
    end

    fprintf('===============================================================\n\n');

end
