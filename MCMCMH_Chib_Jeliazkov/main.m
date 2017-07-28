clear; close all; clc;
global H T

A = [1  0 0
    .5 1 0
    0 0.3  1
    .3 .5  .2
    .2  .9 .7
    0 .6 -.5];
[m, n] = size(A);
T = 200;
isConstrained = false(size(A)); isConstrained(1, 2) = true;
isConstrained(1:2, 3) = true;
%isConstrained(3, 2) = true;

%% Generate data
A0 = A(1:3, :); A1 = A(4:6, :);
resid = randn(T, 3);
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
A0 = A0 + reshape(randn(1, m*n), m, n);

fprintf('True model %d; Testing model %d', 10, 10)
tic
[Ahat, Accept_ratio, acceptanceRate] = mcmc4(isConstrained, A0);
toc

mcmc_A = mcmc_summary(Ahat', 500, 1, 1);
benchmark = reshape(A, [], 1);
xbins = linspace(-5, 5, 200);
plot_mcmc_hist(mcmc_A , xbins, [3, 3], benchmark)

mean = reshape(mcmc_A.descr_stats(:, 1), 6, 3);
q025 = reshape(mcmc_A.quantiles(:, 2), 6, 3);
q975 = reshape(mcmc_A.quantiles(:, 10), 6, 3);

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

 
 disp('Acceptance rate:')
 disp(acceptanceRate)