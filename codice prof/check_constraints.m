function check_constraints(w, table, portfolio_name)
    cyclical = contains(table.MacroGroup, 'Cyclical');
    defensive = contains(table.MacroGroup, 'Defensive');
    neutral = contains(table.MacroGroup, 'Neutral');
    
    fprintf('\n=== %s ===\n', portfolio_name);
    fprintf('Defensive: %.2f%% (limite: ≤45%%)\n', sum(w(defensive)) * 100);
    fprintf('Neutral:  %.2f%% (limite: ≥20%%)\n', sum(w(neutral)) * 100);
    fprintf('Cyclical: %.2f%%\n', sum(w(cyclical)) * 100);
    fprintf('Max weight: %.2f%% (limite: 30%%)\n', max(w) * 100);
    fprintf('Sum weights: %.6f\n', sum(w));
    
    % Check vincoli
    constraints_ok = true;
    if sum(w(defensive)) > 0.45
        fprintf('NO Defensive > 45%%\n');
        constraints_ok = false;
    end
    if sum(w(neutral)) < 0.20
        fprintf('NO Neutral < 20%%\n');
        constraints_ok = false;
    end
    if max(w) > 0.30
        fprintf('NO Max weight > 30%%\n');
        constraints_ok = false;
    end
    
    if constraints_ok
        fprintf('TUTTI I VINCOLI RISPETTATI\n');
    end
end