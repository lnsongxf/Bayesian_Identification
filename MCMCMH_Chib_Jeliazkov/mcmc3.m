function [params, Accept_ratio, acceptRate] = mcmc3(isConstrained, A0)
nBurn = 500; nDraw = 1e4; N = nBurn + nDraw;
%sdPrecised = 0.002; sdNotPrecised = 200;
% sdNotPrecised = 200;
sdProp = 0.15;

[m, n] = size(isConstrained);
% restr_seq = indx(isConstrained == 1);
% unrestr_seq = indx(isConstrained == 0);

params = zeros(N, m * n);
params(1, :) = A0(:);
nAccepted = zeros(m ,n);
Accept_ratio = ones(N, n*m);
for i_rep = 2:N
    Aold = reshape(params(i_rep - 1, :), size(isConstrained));
    %Sample A0
    k = 1;
    for j = 1:n
        for i = 1:m
            Aprop = Aold;
            if (isConstrained(i, j))
                Aold(i, j) = 0;
            else
                Aprop(i, j) = Aprop(i, j) + randn(1) * sdProp;
                
                Accept_ratio(i_rep, k) = exp(nll(Aold) - nll(Aprop));
                 isAccepted = rand(1) <= Accept_ratio(i_rep, k);
                 if isAccepted
                     nAccepted(i, j) = nAccepted(i, j) + 1;
                     Aold = Aprop;
                 end
            end
             k = k + 1;
        end
    end
     
    params(i_rep, :) = Aold(:);
end
acceptRate = nAccepted / N;
params = params(nBurn + 1:end, :);


