function Print_Portfolio_Comparison(DR_eq, DR_MDR, DR_ME, ...
                                    Vol_eq, Vol_MDR, Vol_ME, ...
                                    SR_eq, SR_MDR, SR_ME, ...
                                    HI_eq, HI_MDR, HI_ME)
% PRINT_PORTFOLIO_COMPARISON  Stampa una tabella comparativa delle metriche dei portafogli
%
% Input:
%   DR_*   : Diversification Ratio
%   Vol_*  : Volatilità
%   SR_*   : Sharpe Ratio
%   HI_*   : Herfindahl Index
%   *_eq   : Equal-weighted portfolio
%   *_MDR  : Maximum Diversification Ratio portfolio
%   *_ME   : Maximum Entropy portfolio

    MetricNames = {'DR'; 'Volatility'; 'Sharpe Ratio'; 'Herfindahl Index'};

    % Matrice 4x3 con tutte le metriche
    MetricsMatrix = [
        DR_eq,   DR_MDR,   DR_ME;
        Vol_eq,  Vol_MDR,  Vol_ME;
        SR_eq,   SR_MDR,   SR_ME;
        HI_eq,   HI_MDR,   HI_ME
    ];

    % Conversione in tabella
    ComparisonTable = array2table(MetricsMatrix, ...
        'RowNames', MetricNames, ...
        'VariableNames', {'EqualWeighted', 'MDR_G', 'ME_H'});

    fprintf('\n=== Comparison of Portfolios: EW vs. MDR (G) vs. ME (H) ===\n');
    disp(ComparisonTable);

end
