function [params, Accept_ratio, acceptRate] = mcmc2(isConstrained, A0)
nBurn = 500; nDraw = 1e4; N = nBurn + nDraw;
sdPrecised = 0.002; sdNotPrecised = 200;
sdProp = 0.05;

[m, n] = size(isConstrained);
indx = reshape(1:(m*n), m, n);
restr_seq = indx(isConstrained == 1);
unrestr_seq = indx(isConstrained == 0);

params = zeros(N, m * n);
params(1, :) = A0(:);
nAccepted = zeros(m ,n);
Accept_ratio = ones(N, 2);
for i_rep = 2:N
     Aold = reshape(params(i_rep - 1, :), size(isConstrained));
     % 1. Restricted block parameters
     % 1.1 Sample restricted parameters
     Aprop = Aold;
          
     for j = restr_seq'
          Aprop(j) = Aprop(j)+ randn(1) * sdProp;
          Accept_ratio(i_rep, 1) = Accept_ratio(i_rep, 1) * normpdf(Aprop(j),...
                0, sdPrecised) /normpdf(Aold(j), 0, sdPrecised);
     end
     % 1.2 Accept sampled restricted parameters
     Accept_ratio(i_rep, 1) = Accept_ratio(i_rep, 1) * ...
        exp(nll(Aold) - nll(Aprop));
     isAccepted = rand(1) <= Accept_ratio(i_rep, 1);
     if isAccepted
         for j = restr_seq'
             nAccepted(j) = nAccepted(j) + 1;
         end
         Aold = Aprop;
     end
     
     % 2. Unrestricted block parameters
     % 1.1 Sample unrestricted parameters
     Aprop = Aold;
     for j = unrestr_seq' 
          Aprop(j) = Aprop(j)+ randn(1) * sdProp;
          Accept_ratio(i_rep, 2) = Accept_ratio(i_rep, 2) * normpdf(Aprop(j),...
                0, sdNotPrecised) /normpdf(Aold(j), 0, sdNotPrecised);
     end
     % 1.2 Accept sampled unrestricted parameters
     Accept_ratio(i_rep, 2) = Accept_ratio(i_rep, 2) * ...
        exp(nll(Aold) - nll(Aprop));
     isAccepted = rand(1) <= Accept_ratio(i_rep, 2);
     if isAccepted
         for j = unrestr_seq' 
             nAccepted(j) = nAccepted(j) + 1;
         end
         Aold = Aprop;
     end
     
    params(i_rep, :) = Aold(:);
end
acceptRate = nAccepted / N;
params = params(nBurn + 1:end, :);


