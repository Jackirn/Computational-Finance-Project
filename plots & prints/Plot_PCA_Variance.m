function Plot_PCA_Variance(ExplainedVar)
% PLOT_PCA_VARIANCE  Plotta la varianza spiegata da ciascun Principal Component in modo pulito.

    figure('Color', [1 1 1]); % sfondo bianco
    bar(ExplainedVar*100, 'FaceColor', [0 0.447 0.741], 'EdgeColor', 'k'); % blu con bordo nero
    grid on;
    
    ax = gca;
    ax.GridLineStyle = '--';
    ax.GridAlpha = 0.7;
    ax.Box = 'on';
    ax.FontSize = 12;
    ax.LineWidth = 1.2;

    title('Variance Explained by Each Principal Component', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Principal Component', 'FontSize', 13, 'FontWeight', 'bold');
    ylabel('Explained Variance (%)', 'FontSize', 13, 'FontWeight', 'bold');

    ylim([0, max(ExplainedVar*100)*1.1]);

    % Etichette sopra le barre
    for i = 1:length(ExplainedVar)
        text(i, ExplainedVar(i)*100 + 1, sprintf('%.1f%%', ExplainedVar(i)*100), ...
            'HorizontalAlignment', 'center', 'FontSize', 11);
    end
    %exportgraphics(gcf, 'Plots/PCA_var.pdf', 'ContentType', 'vector')
end
