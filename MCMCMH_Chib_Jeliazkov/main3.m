clear; close all; clc;
global H T

% A = [1  0 0
%     .5 1 0
%     0 0.3  1
%     .3 .5  .2
%     .2  .9 .7
%     0 .6 -.5];
A = [1  0.7
    .5 1 
    0 0 
    0 0.4 ];

[m, n] = size(A);
T = 200;
isConstrained = false(size(A));
isConstrained(3, 2) = true;
isConstrained(3, 1) = true;
isConstrained(4, 1) = true;
%isConstrained = false(size(A)); isConstrained(1, 2) = true;
%isConstrained(1:2, 3) = true;
%isConstrained(3, 2) = true;

%% Generate data
A0 = A(1:n, :); A1 = A((n+1):m, :);
resid = randn(T, n);
Y(1, :) = resid(1, :) / A0;
for i = 2:T
    Y(i, :) = (-Y(i - 1, :) * A1 + resid(i, :))/ A0;
end

Z = Y(1:(end - 1), :); Y = Y(2:end, :);

X = [Y Z];

H = chol(cov(X));
%% MH MCMC

[A0, isIdentified] = tsls(X, isConstrained);
if any(~isIdentified)
    error('The structural model is not identified')
end
which_sample = ones(m, n) == 1;
fprintf('MCMCMH procedure starts \n')
tic
[Ahat, Accept_ratio, acceptanceRate] = mcmc_bad2(isConstrained, A0, ...
    which_sample, 500, 1e4);
toc


mcmc_A = mcmc_summary(Ahat', 500, 1, 0);
benchmark = reshape(A, [], 1);
xbins = linspace(-5, 5, 200);
plot_mcmc_hist(mcmc_A , xbins, [n, n], benchmark)

mean = reshape(mcmc_A.descr_stats(:, 1), m, n);
q025 = reshape(mcmc_A.quantiles(:, 2), m, n);
q975 = reshape(mcmc_A.quantiles(:, 10), m, n);

fprintf('Chib procedure starts \n')
% log_ml = newton_raftery_marginal_ll(Ahat, n, m);
tic;
log_ml = chib_marginal_ll(mean, Ahat, Y, Z, isConstrained,  500, 1e3);
toc

fprintf('Alternative model check. MCMCMH procedure starts \n')
isConstrained2 = isConstrained;
isConstrained2(2, 1) = true;
tic
[Ahat, Accept_ratio, acceptanceRate] = mcmc_bad2(isConstrained2, A0, ...
    which_sample, 500, 1e3);
toc
mcmc_A = mcmc_summary(Ahat', 500, 1, 0);
mean2 = reshape(mcmc_A.descr_stats(:, 1), m, n);


fprintf('Alternative model check. Chib procedure starts \n')
tic;
log_ml2 = chib_marginal_ll(mean2, Ahat, Y, Z, isConstrained2,  500, 1e3);
toc

disp('Mean Estimate:')
disp(mean)

disp('2.5 % quantile:')
disp(q025)

disp('97.5 % quantile:')
disp(q975)

disp('Real Parameters:')
disp(A)


disp('Theoretical Concentration Matrix:')
disp(A * A')

  disp('Marginal Likelihood:')
 disp(log_ml)
 
   disp('Marginal Likelihood of alternative model:')
 disp(log_ml2)
 
  disp('Posterior odds:')
  PO = exp(log_ml2 - log_ml);
  disp(PO)
  
  disp('Probability of the true model:')
  disp(1/(PO + 1))