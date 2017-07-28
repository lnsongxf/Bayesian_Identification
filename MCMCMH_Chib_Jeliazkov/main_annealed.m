clear; close all; clc;
global H T

% A = [1  0 0
%     .5 1 0
%     0 0.3  1
%     .3 .5  .2
%     .2  .9 3.7
%     0 .6 -.5];
A = [1  0.7
    .5 1 
    0.3 0.3 
    0.5 0.4 ];
true_model = 11;

[m, n] = size(A);
T = 200;
num_tries = 100;

true_restr = decode_restr(true_model, n, m-1) == 0;
true_restr  = [true_restr(1:n, :); ones(1, n) == 1; ...
        true_restr(end, :)];
%isConstrained = false(size(A)); isConstrained(1, 2) = true;
%isConstrained(1:2, 3) = true;
%isConstrained(3, 2) = true;

%% Generate data
A_true = A .* (~true_restr) ;
A0 = A_true(1:n, :); A1 = A_true((n+1):m, :);
resid = randn(T, n);
Y(1, :) = resid(1, :) / A0;
for i = 2:T
    Y(i, :) = (-Y(i - 1, :) * A1 + resid(i, :))/ A0;
end

Z = Y(1:(end - 1), :); Y = Y(2:end, :);

X = [Y Z];

H = chol(cov(X));
%% MH MCMC
log_ml = zeros(num_tries, 15);log_ml2 = zeros(num_tries, 15);
var_ml = zeros(num_tries, 15);
mcmc_A = cell(1, num_tries); mean_A = cell(1, num_tries);
Prob = zeros(num_tries, 15); Prob2 = zeros(num_tries, 15);
prior_vars = [0.002, 200];
prior_var = ones(m, n)* 200;
beta_seq = 0:0.1:1;

for s = 1:num_tries
    mcmc_A{s} = cell(1, 15); mean_A{s} = cell(1, 15);
    for model = 1:15
        fprintf(['Try ', num2str(s), '; Model ', num2str(model), '\n'])
        isConstrained = decode_restr(model, n, m-1) == 0;
        isConstrained = [isConstrained(1:n, :); ones(1, n) == 1; ...
            isConstrained(end, :)];
        [A0, isIdentified] = tsls(X, isConstrained);
        if any(~isIdentified)
            error('The structural model is not identified')
        end

        which_sample = ones(m, n) == 1;
        fprintf('True model %f; Test model %f; AIS procedure starts \n', true_model, model)
        tic
        [Ahat,  Accept_ratio, acceptRate, log_w, lml] =...
            annealed_imp_sample(prior_var, beta_seq, isConstrained, ...
            A0, which_sample, 500, 1e3);
        toc
        
        lml(isnan(lml)) = -Inf;
        log_ml(s, model) = lml;
    end
    Prob(s, :) = exp(log_ml(s, :))./sum(exp(log_ml(s, :)));
    Prob2(s, :) = exp(log_ml2(s, :))./sum(exp(log_ml2(s, :)));
end

disp('Marginal Likelihoods (AIS method):')
disp(log_ml)

  disp('Posterior probabilities (AIS method):')
 disp(mean(Prob, 1))
 
 bar(mean(Prob, 1))
 title(['AIS method; True model: ', num2str(true_model), '; Number of resamples: ', num2str(100)])
 
 save 'ais_model_11.mat'