function [params,  Accept_ratio, acceptRate] = mcmc_bad2(isConstrained, A0, which_sample, nBurn, nDraw)
% Performs MCMCMH algorithm to sample parameter values
% for the SVAR model in the form:
% YA = ZB + UD^(-0.5)
%
% Nickolay Arefiev (n.arefiev@gmail.com)
% Ramis Khabibullin (rawirtschaft@gmail.com)
N = nBurn + nDraw;
sdProp = 0.25;

[m, n] = size(isConstrained);
indx =  reshape(1:(m*n), m, n);
% restr_seq = indx(isConstrained == 1);
% unrestr_seq = indx(isConstrained == 0);

D0 = diag(A0).^2;
A0 = A0 / diag(sqrt(D0));
A0(isConstrained) = 0;

Ddraws = zeros(N, n);
params = zeros(N, m * n);
params(1, :) = A0(:);
Ddraws(1, :) = D0(:);
nAccepted = zeros(m ,n);
Accept_ratio = ones(N, n*m);

for i_rep = 2:N
    Dold = Ddraws(i_rep-1, :)';
    Aold = reshape(params(i_rep - 1, :), size(isConstrained));
    %Sample A0
    for j = 1:n
        Aold(j, j) = 1;
    end
    for j = 1:n
        Dprop = Dold;
        Dprop(j) = (sqrt(Dprop(j)) + randn(1) * sdProp).^2;
        
        Accept_ratio(i_rep, indx(j, j)) = ...
            exp(nll_bad(Aold, Dold) - nll_bad(Aold, Dprop));
        isAccepted = rand(1) <= Accept_ratio(i_rep, indx(j, j));
        if isAccepted
           nAccepted(j, j) = nAccepted(j, j) + 1;
           Dold = Dprop;
        end
        for i = 1:m
            Aprop = Aold;
            if ((~isConstrained(i, j))&&(which_sample(i , j) == 1)&&(i ~= j))
                Aprop(i, j) = Aprop(i, j) + randn(1) * sdProp;
                
                Accept_ratio(i_rep, indx(i, j)) = ...
                    exp(nll_bad(Aold, Dold) - nll_bad(Aprop, Dold));
                isAccepted = rand(1) <= Accept_ratio(i_rep, indx(i, j));
                if isAccepted
                    nAccepted(i, j) = nAccepted(i, j) + 1;
                    Aold = Aprop;
                end
            end
        end
    end
    for j = 1:n
        Aold(j, j) = sqrt(Dold(j));
    end
    params(i_rep, :) = Aold(:);
    Ddraws(i_rep, :) = Dold(:);
end
acceptRate = nAccepted / N;
params = params(nBurn + 1:end, :);
end


