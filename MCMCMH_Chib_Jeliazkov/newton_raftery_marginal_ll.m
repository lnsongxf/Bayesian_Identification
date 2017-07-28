function [log_ml] = newton_raftery_marginal_ll(params, n, m)
% Calculates marginal likelihood by harmonic mean method
%
% Ramis Khabibullin (rawirtschaft@gmail.com)
    N = size(params, 1);
    log_lik_seq = zeros(N, 1);
    for i_rep = 1:N
        A = reshape(params(i_rep, :), m, n);
        D = diag(A).^2;
        A = A / diag(sqrt(D));
        
        log_lik_seq(i_rep, :) = exp(nll_bad(A, D));
    end
    log_ml = -log(mean(log_lik_seq));
end