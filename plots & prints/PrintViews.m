function PrintViews(q)
    fprintf('\n=============================================\n');
    fprintf('        Black–Litterman Views Check          \n');
    fprintf('=============================================\n');
    
    fprintf('View 1:  Cyclical   >  Neutral        by %+5.2f%%\n', q(1)*100);
    fprintf('View 2:  Asset 10   <  Defensive avg  by %+5.2f%%\n', q(2)*100);
    fprintf('View 3:  Asset 2    >  Asset 13       by %+5.2f%%\n', q(3)*100);
    
    fprintf('---------------------------------------------\n\n');
end

