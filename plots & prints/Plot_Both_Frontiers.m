function Plot_Both_Frontiers(FrontierVola, FrontierRet, meanRisk, meanRet, ...
                              w_MVP, w_MSRP, w_MVP_RF, w_MSRP_RF, V, ExpRet, rf)
%PLOT_BOTH_FRONTIERS Plots comparison between Standard and Robust frontiers
%   Only the efficient branch (R >= MVP) is shown

    % Ensure column vector for ExpRet
    ExpRet = ExpRet(:);

    % --- Find MVP points ---
    [~, idx_MVP_std] = min(FrontierVola);
    ret_MVP_std = FrontierRet(idx_MVP_std);
    
    [~, idx_MVP_rob] = min(meanRisk);
    ret_MVP_rob = meanRet(idx_MVP_rob);

    % --- Filter only efficient points ---
    efficient_std = FrontierRet >= ret_MVP_std;
    efficient_rob = meanRet >= ret_MVP_rob;

    % Standard portfolios (A, B)
    vol_MVP  = sqrt(w_MVP'  * V * w_MVP);
    ret_MVP  = w_MVP'  * ExpRet;
    vol_MSRP = sqrt(w_MSRP' * V * w_MSRP);
    ret_MSRP = w_MSRP' * ExpRet;
    
    % Robust portfolios (C, D)
    vol_MVP_RF  = sqrt(w_MVP_RF'  * V * w_MVP_RF);
    ret_MVP_RF  = w_MVP_RF'  * ExpRet;
    vol_MSRP_RF = sqrt(w_MSRP_RF' * V * w_MSRP_RF);
    ret_MSRP_RF = w_MSRP_RF' * ExpRet;
    
    figure('Position', [100, 100, 1000, 700]);
    
    % Plot Standard Frontier (efficient branch)
    plot(FrontierVola(efficient_std)*100, FrontierRet(efficient_std)*100, 'LineWidth', 2.5, ...
         'Color', [0 0.45 0.74], 'DisplayName', 'Standard Frontier'); % blu
    hold on;
    
    % Plot Robust Frontier (efficient branch)
    plot(meanRisk(efficient_rob)*100, meanRet(efficient_rob)*100, 'LineWidth', 2.5, ...
         'Color', [0.85 0.33 0.10], 'DisplayName', 'Robust Frontier'); % arancione
    
    % Plot portfolios as before
    scatter(vol_MVP*100, ret_MVP*100, 120, 'filled', ...
            'MarkerFaceColor', [0 0.45 0.74], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Portfolio A (Standard MVP)');
    scatter(vol_MSRP*100, ret_MSRP*100, 120, 's', 'filled', ...
            'MarkerFaceColor', [0 0.45 0.74], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Portfolio B (Standard MSRP)');
    
    scatter(vol_MVP_RF*100, ret_MVP_RF*100, 120, 'filled', ...
            'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Portfolio C (Robust MVP)');
    scatter(vol_MSRP_RF*100, ret_MSRP_RF*100, 120, 's', 'filled', ...
            'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Portfolio D (Robust MSRP)');
    
    % Add CAL lines (unchanged)
    x_max = max([FrontierVola(:); meanRisk(:)]) * 100 * 1.1;
    x_line = [0, x_max];
    
    rf_plot = rf * 100;  
    sharpe_std = (ret_MSRP - rf) / vol_MSRP;
    sharpe_rf  = (ret_MSRP_RF - rf) / vol_MSRP_RF;
    
    y_CAL_std = rf_plot + sharpe_std * x_line;
    y_CAL_rf  = rf_plot + sharpe_rf  * x_line;
    
    plot(x_line, y_CAL_std, '--', 'LineWidth', 1.5, ...
         'Color', [0 0.45 0.74], 'DisplayName', 'CAL (Standard)');
    plot(x_line, y_CAL_rf, '--', 'LineWidth', 1.5, ...
         'Color', [0.85 0.33 0.10], 'DisplayName', 'CAL (Robust)');
    
    xlabel('Volatility (%)', 'FontSize', 12);
    ylabel('Expected Return (%)', 'FontSize', 12);
    title('Efficient Frontier: Standard vs Robust (Resampling)', ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    box on;
    set(gca, 'FontSize', 11);
    hold off;

    % exportgraphics(gcf, 'Plots/entrambe.pdf', 'ContentType', 'vector');
end
