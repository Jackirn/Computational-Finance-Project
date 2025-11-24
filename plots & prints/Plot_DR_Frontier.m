function Plot_DR_Frontier(target_Vol, DR_vals)
% PLOT_DR_FRONTIER  Plotta la frontiera DR vs Volatilità.

    figure;
    plot(target_Vol, DR_vals, 'LineWidth', 2);

    xlabel('Portfolio Volatility', 'FontSize', 12);
    ylabel('Diversification Ratio', 'FontSize', 12);
    title('Diversification–Risk Frontier (Exercise 3.a)', 'FontSize', 14);

    grid on;

end
