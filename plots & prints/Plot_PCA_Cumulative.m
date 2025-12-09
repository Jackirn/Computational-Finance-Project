function Plot_PCA_Cumulative(n_pc, CumExplVar)
% PLOT_PCA_CUMULATIVE  Plotta la varianza cumulativa spiegata dai Principal Components
% con linea orizzontale al 85%.

    figure;
    plot(n_pc, CumExplVar, 'm', 'LineWidth', 2);
    hold on;
    yline(80, '--r', '80% Threshold', 'LineWidth', 1.5, 'LabelHorizontalAlignment','left', 'FontSize',12);
    scatter(n_pc, CumExplVar, 'm', 'filled');
    grid on;
    xlabel('Number of Principal Components');
    ylabel('Cumulative Explained Variance (%)');
    title('Cumulative Percentage of Explained Variance');
    hold off;
    %exportgraphics(gcf, 'Plots/PCA_cum.pdf', 'ContentType', 'vector')
end
