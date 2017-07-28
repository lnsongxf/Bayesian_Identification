function [A, isIdentified] = tsls(X, isConstrained)
% Estimates Structural VAR using two-stage least squares procedure

    function ret = parents(nodeIndex)
        ret = X(:, setdiff(find(~isConstrained(:, nodeIndex)), nodeIndex));
    end
    function ret = instruments
        ret = [X(:, (n + 1):end), residuals(:, isIdentified)];
    end
    function [isEqIdentified, equation] = estimateEquation(equationNdx)
        par = detrend(parents(equationNdx), 'constant');
        instr = detrend(instruments, 'constant');
        node = X(:, equationNdx);
        Pi = (instr' * instr) \ (instr' * par);
        pi = (instr' * instr) \ (instr' * node);
        if size(par, 2) == 0
            isEqIdentified = true;
        else
            isEqIdentified = sum(abs(eig(Pi' * Pi)) > tolerance) >= size(par, 2);
        end
        if ~isEqIdentified
            equation = NaN;
            return;
        end
        if size(par, 2) > 0
            coefs = (Pi' * Pi) \ Pi' * pi;
            eqResiduals = node - par * coefs;
        else
            coefs = zeros(0);
            eqResiduals = node;
        end
        
        equation = zeros(size(isConstrained, 1), 1);
        counter = 0;
        for i1 = 1:length(equation)
            if i1 == equationNdx
                equation(i1) = 1;
                continue
            end
            if isConstrained(i1, equationNdx)
                continue
            end
            counter = counter + 1;
            equation(i1) = -coefs(counter);
        end
        equation = equation ./ std(eqResiduals);
    end
tolerance = 1e-5;
n = size(isConstrained, 2);
T = size(X, 1);
isIdentified = false(1, n);
residuals = NaN(T, n);
A = NaN(size(isConstrained));
newNodeHasBeenIdentified = true;
while newNodeHasBeenIdentified
    newNodeHasBeenIdentified = false;
    for i = 1:n
        if isIdentified(i)
            continue
        end
        [isEqIdentified, equation] = estimateEquation(i);
        if isEqIdentified
            newNodeHasBeenIdentified = true;
            A(:, i) = equation;
            residuals(:, i) = X * A(:, i);
            isIdentified(i) = true;
        end
    end
end
end

