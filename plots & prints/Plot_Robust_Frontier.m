function Plot_Robust_Frontier(meanRisk, meanRet, NumAssets, V, ExpRet, w_MVP_RF, w_MSRP_RF, rf)
%PLOT_ROBUST_ONLY Plots only the Robust Efficient Frontier and key portfolios
%
%   Inputs:
%       meanRisk, meanRet : Robust frontier (std and mean returns)
%       NumAssets         : Number of assets
%       V, ExpRet         : Covariance matrix and expected returns
%       w_MVP_RF          : Robust Minimum Variance Portfolio weights
%       w_MSRP_RF         : Robust Maximum Sharpe Ratio Portfolio weights
%       rf                : Risk-free rate

    % EW
    w_EW = (1/NumAssets) * ones(NumAssets,1);
    vol_EW = sqrt(w_EW' * V * w_EW);
    ret_EW = w_EW' * ExpRet';

    % Robust MVP
    vol_MVP_RF = sqrt(w_MVP_RF' * V * w_MVP_RF);
    ret_MVP_RF = w_MVP_RF' * ExpRet';

    % Robust MSRP
    vol_MSRP_RF = sqrt(w_MSRP_RF' * V * w_MSRP_RF);
    ret_MSRP_RF = w_MSRP_RF' * ExpRet';

    sharpe_RF = (ret_MSRP_RF - rf) / vol_MSRP_RF;

    figure('Position', [100, 100, 900, 650]);
    plot(meanRisk, meanRet, 'LineWidth', 2.5, 'Color', [0.85 0.33 0.10], ...
         'DisplayName', 'Robust Efficient Frontier'); % arancione elegante
    hold on;

    % Scatter dei portafogli chiave
    scatter(vol_EW, ret_EW, 100, 'filled', ...
            'MarkerFaceColor', [0 0.45 0.74], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.2, 'DisplayName', 'Equal Weight');
    scatter(vol_MVP_RF, ret_MVP_RF, 120, 'filled', ...
            'MarkerFaceColor', [0.47 0.67 0.19], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Portfolio C (Robust MVP)');
    scatter(vol_MSRP_RF, ret_MSRP_RF, 120, 's', 'filled', ...
            'MarkerFaceColor', [0.93 0.69 0.13], 'MarkerEdgeColor', 'k', ...
            'LineWidth', 1.5, 'DisplayName', 'Portfolio D (Robust MSRP)');

    % CAL (Capital Allocation Line) 
    x_max = max(meanRisk) * 1.1;
    x_line = [0, x_max];
    y_CAL = rf + sharpe_RF * x_line;
    plot(x_line, y_CAL, '--', 'Color', [0.85 0.33 0.10], ...
         'LineWidth', 1.5, 'DisplayName', 'CAL (Robust)');

    xlabel('Volatility (σ)', 'FontSize', 12);
    ylabel('Expected Return (μ)', 'FontSize', 12);
    title('Robust Efficient Frontier and Key Portfolios', ...
          'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 11);
    grid on; box on;
    set(gca, 'FontSize', 11);
    hold off;

end