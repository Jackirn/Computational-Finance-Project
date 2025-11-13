function [c, ceq] = nonlinConstr(x, V, vol_i)
    c = [];
    ceq = sqrt(x'*V*x)-vol_i;
end