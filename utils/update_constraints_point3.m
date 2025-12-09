function new_constr = update_constraints_point3(old_constr, groups)
% UPDATE_CONSTRAINTS_POINT3 Updates portfolio constraints for the Alternative Risk exercise.
%
%   new_constr = UPDATE_CONSTRAINTS_POINT3(old_constr, groups) modifies the
%   constraint structure to meet the specific requirements of Point 3 
%   (Maximum Diversification and Entropy):
%     1. Individual Asset Upper Bound (ub): 25%
%     2. Cyclical Group Weight: >= 20%
%     3. Defensive Group Weight: <= 50%
%
%   INPUTS:
%     old_constr - Struct containing previous constraint matrices (A, b, Aeq, beq, lb, ub).
%     groups     - Struct containing logical vectors for asset groups (cyclical, defensive, etc.).
%
%   OUTPUT:
%     new_constr - Updated constraint structure ready for fmincon optimization.

    NumAssets = length(old_constr.lb);
    new_constr = old_constr;
    
    % Update Bounds
    % Max 25% per single asset, Min 0%
    new_constr.ub = 0.25 * ones(NumAssets, 1);
    new_constr.lb = zeros(NumAssets, 1);
    
    % Define Group Constraints Logic
    % Requirement A: Cyclical >= 20%  -->  -sum(w_cyc) <= -0.20 (Linear Inequality form)
    % Requirement B: Defensive <= 50% -->   sum(w_def) <= 0.50
    
    % Ensure logical vectors are transposed to row vectors (1 x N)
    % Note: 'groups.cyclical' comes as (N x 1), so we transpose it.
    A_cyc = -double(groups.cyclical'); 
    A_def =  double(groups.defensive');
    
    b_cyc = -0.20;
    b_def = 0.50;
    
    % Construct Linear Inequality Matrix (A * x <= b)
    % Vertical Concatenation: 
    % Row 1: Cyclical constraint
    % Row 2: Defensive constraint
    % Rows 3 to N+2: Individual upper bounds (enforced via Identity matrix)
    new_constr.A = [A_cyc; A_def; eye(NumAssets)]; 
    new_constr.b = [b_cyc; b_def; new_constr.ub];
    
    % Equality Constraints
    % Fully invested constraint: sum(w) = 1
    new_constr.Aeq = ones(1, NumAssets);
    new_constr.beq = 1;
end