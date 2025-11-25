function Print_Ptfs(ret,vol,w,name,flag,SR)
    fprintf('\n===============================================================\n');
    fprintf('                     Portfolio %-10s\n', name);
    fprintf('===============================================================\n');

    % Core stats
    fprintf('Expected Return:      %8.3f%%\n', ret * 100);
    fprintf('Volatility:           %8.3f%%\n', vol * 100);

    if (flag == 1)
        fprintf('Sharpe Ratio:         %8.3f\n', SR);
    end

    % Top holdings
    fprintf('---------------------------------------------------------------\n');
    fprintf('Top 3 Holdings:\n');

    [sorted_w, idx] = sort(w, 'descend');

    for i = 1:min(3, length(w))
        if sorted_w(i) > 0.005   % show only meaningful weights (>0.5%)
            fprintf('  %-12s   %8.2f%%\n', ...
                sprintf('Asset %d', idx(i)), sorted_w(i)*100);
        end
    end

    fprintf('===============================================================\n\n');

end

