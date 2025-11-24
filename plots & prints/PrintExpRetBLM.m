function PrintExpRetBLM(NumAssets,mu_BL,mu_MKT,nm)
    fprintf('\n===============================================================\n');
    fprintf('            Black–Litterman Posterior: Return Shifts           \n');
    fprintf('===============================================================\n');
    fprintf('Asset                 Prior       Posterior     Change     %% Change\n');
    fprintf('---------------------------------------------------------------\n');
    
    changes = mu_BL - mu_MKT;
    [~, sorted_idx] = sort(abs(changes), 'descend');
    
    for i = 1:min(8, NumAssets)
        idx = sorted_idx(i);
    
        % Avoid division by zero in pct change
        if abs(mu_MKT(idx)) < 1e-8
            pct_change = NaN;
        else
            pct_change = (changes(idx) / abs(mu_MKT(idx))) * 100;
        end
    
        fprintf('%-15s   %+8.4f    %+8.4f    %+8.4f    %+7.2f%%\n', ...
                nm{idx}, mu_MKT(idx), mu_BL(idx), changes(idx), pct_change);
    end
    
    view_impact = norm(mu_BL - mu_MKT);
    
    fprintf('---------------------------------------------------------------\n');
    fprintf('Total deviation induced by views: %.6f\n', view_impact);
    fprintf('Relative magnitude vs avg prior:   %.2f%%\n', ...
            (view_impact / mean(abs(mu_MKT))) * 100);
    fprintf('===============================================================\n\n');

end

