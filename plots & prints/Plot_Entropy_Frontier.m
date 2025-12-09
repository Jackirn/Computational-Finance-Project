function Plot_Entropy_Frontier(target_Vol, Entropy_vals)
% PLOT_ENTROPY_FRONTIER  Plotta la frontiera Entropy vs Volatilità.

    figure;
    plot(target_Vol, Entropy_vals, 'LineWidth', 2);

    xlabel('Portfolio Volatility', 'FontSize', 12);
    ylabel('Entropy in risk contributions', 'FontSize', 12);
    title('Entropy–Risk Frontier', 'FontSize', 14);

    grid on;
    %exportgraphics(gcf, 'Plots/div_entr.pdf', 'ContentType', 'vector')
end
