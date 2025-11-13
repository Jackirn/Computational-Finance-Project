function Plot_Frontier(FrontierVola,FrontierRet,NumAssets,V,ExpRet,SharpeFrontier)
%PLOT_FRONTIER Summary of this function goes here
%   Detailed explanation goes here

    % EW
    WeightsEW = (1/NumAssets).*ones(NumAssets,1);
    vol_EW = sqrt(WeightsEW'*V*WeightsEW);
    ret_EW = WeightsEW'*ExpRet';
    
    % MV
    [~, idx_MVP] = min(FrontierVola);
    vol_MVP = FrontierVola(idx_MVP);
    ret_MVP = FrontierRet(idx_MVP);
    
    % MS
    [~, idx_MSRP] = max(SharpeFrontier);
    vol_MSRP = FrontierVola(idx_MSRP);
    ret_MSRP = FrontierRet(idx_MSRP);
    
    figure;
    plot(FrontierVola, FrontierRet, 'LineWidth', 2.5, 'Color', [0 0.45 0.74]); % blu elegante
    hold on;
    
    scatter(vol_EW, ret_EW, 80, 'filled', 'MarkerFaceColor', [0.85 0.33 0.10]); % arancione
    scatter(vol_MVP, ret_MVP, 80, 'filled', 'MarkerFaceColor', [0.47 0.67 0.19]); % verde
    scatter(vol_MSRP, ret_MSRP, 80, 'filled', 'MarkerFaceColor', [0.93 0.69 0.13]); % giallo
    
    legend({'Efficient Frontier', 'Equal Weight (EW)', 'Minimum Variance (MVP)', 'Maximum Sharpe (MSRP)'}, ...
        'Location', 'best', 'FontSize', 11);
    
    xlabel('Volatility (σ)', 'FontSize', 12);
    ylabel('Expected Return (μ)', 'FontSize', 12);
    title('Efficient Frontier and Key Portfolios', 'FontSize', 14, 'FontWeight', 'bold');
    
    grid on;
    box on;
    set(gca, 'FontSize', 11);

end

