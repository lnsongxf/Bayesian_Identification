function [params,  Accept_ratio, acceptRate, log_w, marginal_ll]  = annealed_imp_sample(prior_var, beta_seq, isConstrained, A0, which_sample, nBurn, nDraw)
% Calculates marginal likelihood by Annealed Importance Sample method
% 
% Ramis Khabibullin (rawirtschaft@gmail.com)
[m, n] = size(isConstrained);

% Create a cell containing functions
prior_fun = cell(m, n);
for j = 1:(m*n)
    prior_fun{j} = @(x) (- 0.5 * (log(2 * pi)  + log(prior_var(j)) + ...
        x.^2/prior_var(j)));
end

% Initialize values
params = cell(length(beta_seq));
Accept_ratio = params; acceptRate = params; 

for k = 1:length(beta_seq)
    fprintf('beta = %f \n', beta_seq(k))
    [params{k},  Accept_ratio{k}, acceptRate{k}] = ...
        mcmc_annealed(isConstrained, A0, which_sample, nBurn, ...
        nDraw, prior_fun, beta_seq(k));
end

log_w = zeros(1, nDraw);
for i = 1:nDraw
    log_w(i) = 0;
    for k = 1:(length(beta_seq) - 1)
        Ai = reshape(params{k}(i, :), m, n);
        Di = diag(Ai).^2;
        Ai = Ai / diag(sqrt(Di));
        
        log_w(i) = log_w(i) +  ...
            (beta_seq(k) - beta_seq(k + 1)) * nll_bad(Ai, Di);
    end
end

% Calculate log marginal likelihood
min_lw = min(log_w);
norm_weights = exp(log_w - min_lw);
if (any(norm_weights > 1e100))
    norm_weights = exp(log_w);
    marginal_ll = log(mean(norm_weights));
else
    marginal_ll = log(mean(norm_weights)) + min_lw;
end
end