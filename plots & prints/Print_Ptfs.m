function Print_Ptfs(ret,vol,w,name)
%PRINT_PTFS Summary of this function goes here
%   Detailed explanation goes here
    fprintf('\nPortfolio %s:\n', name);
    fprintf('  Expected Return: %.2f%%\n', ret * 100);
    fprintf('  Volatility:      %.2f%%\n', vol * 100);
    fprintf('  Top 3 holdings:\n');
    [sorted_weights_C, idx_C] = sort(w, 'descend');
    for i = 1:3
        if sorted_weights_C(i) > 0.01
            fprintf('    Asset %d: %.2f%%\n', idx_C(i), sorted_weights_C(i)*100);
        end
    end
    fprintf('\n')
end

