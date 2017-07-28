function ret = nll(A)
% Uses globally defined variable H defined as chol(cov(Data))
%
% Returns the negative log likelihood given the matrix of parameters A and 
% the number of observations T
%
% Nickolay Arefiev (n.arefiev@gmail.com)

global H T
n = size(A, 2);
ret = T * (0.5 * n * log(2 * pi)...
    - log(abs(det(A(1:n, :)))) + .5 * sum(sum((H * A).^2)));
end

