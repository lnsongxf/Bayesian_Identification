function [params,  Accept_ratio, acceptRate] = mcmc_annealed(isConstrained, A0, which_sample, nBurn, nDraw, prior_fun, beta)
N = nBurn + nDraw;
%sdPrecised = 0.002; sdNotPrecised = 200;
% sdNotPrecised = 200;
sdProp = 0.25;

[m, n] = size(isConstrained);
indx =  reshape(1:(m*n), m, n);
% restr_seq = indkx(isConstrained == 1);
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
            exp(prior_fun{j, j}(sqrt(Dprop(j))) - prior_fun{j, j}(sqrt(Dold(j))) + ...
        (1 - beta) * (nll_bad(Aold, Dold) - nll_bad(Aold, Dprop)));
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
                    exp(prior_fun{j, j}(Aprop(i, j)) - prior_fun{j, j}(Aold(i, j)) + ...
                (1 - beta) * (nll_bad(Aold, Dold) - nll_bad(Aprop, Dold)));
                isAccepted = rand(1) <= Accept_ratio(i_rep, indx(i, j));
                if isAccepted
                    nAccepted(i, j) = nAccepted(i, j) + 1;
                    Aold = Aprop;
                end
            end
        end
    end
     
    % Sample A+ parameters
%     for j = 1:n 
%         Adraw = zeros(n-m, 1);
%         Adraw(post{j}.restr, :) = mvnrnd(post{j}.P * Aold(1:n , j), ...
%             post{j}.H);
%         Aold((n + 1):m, j) = Adraw;
%     end
%     
%     for j = 1:n
%         for i = (n+1):m
%             Aprop = Aold;
%             if (~isConstrained(i, j))
%                 Aprop(i, j) = Aprop(i, j) + randn(1) * sdProp;
%                 
%                 Accept_ratio(i_rep, k) = exp(nll(Aold) - nll(Aprop));
%                 isAccepted = rand(1) <= Accept_ratio(i_rep, k);
%                 if isAccepted
%                     nAccepted(i, j) = nAccepted(i, j) + 1;
%                     Aold = Aprop;
%                 end
%             end
%              k = k + 1;
%         end
%     end
    for j = 1:n
        Aold(j, j) = sqrt(Dold(j));
    end
    params(i_rep, :) = Aold(:);
    Ddraws(i_rep, :) = Dold(:);
end
acceptRate = nAccepted / N;
params = params(nBurn + 1:end, :);
end


