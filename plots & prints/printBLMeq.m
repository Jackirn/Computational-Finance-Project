function printBLMeq(ExpRet_MKT,sigma2_MKT,lambda,NumAssets,w_MKT,mu_MKT)

    fprintf('\nBLACK-LITTERMAN EQUILIBRIUM RETURNS\n');
    fprintf('====================================\n\n');
    
    fprintf('Market Stats: Return=%.2f%%, Vol=%.2f%%, λ=%.3f\n\n', ...
        ExpRet_MKT*100, sqrt(sigma2_MKT)*100, lambda);
    
    fprintf('Asset     Weight%%   Return%%\n');
    fprintf('───────   ───────   ───────\n');
    for i = 1:NumAssets
        if i <= 9
            fprintf('Asset_0%d   %6.2f    %6.2f\n', i, w_MKT(i)*100, mu_MKT(i)*100);
        else
            fprintf('Asset_%d    %6.2f    %6.2f\n', i, w_MKT(i)*100, mu_MKT(i)*100);
        end
    end
    fprintf('────────   ───────   ───────\n');
    fprintf('TOTAL      %6.2f    %6.2f\n', sum(w_MKT)*100, mean(mu_MKT)*100);

end