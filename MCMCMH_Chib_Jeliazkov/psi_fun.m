function ret = psi_fun(i, alpha, beta, gamma)

if i == 0
    ret = sqrt(pi) * normcdf(gamma * sqrt(2));
elseif i == 1
    ret = - 0.5 * alpha * exp( - gamma^2) - beta * psi_fun(0, alpha, beta, gamma);
else
    ret = 0.5*(i - 1)*alpha^2*psi_fun(i - 2, alpha, beta, gamma) - ...
        beta*psi_fun(i - 1, alpha, beta, gamma) - ...
        0.5*alpha*(alpha * gamma - beta)^(i - 1) * exp(- gamma^2);
end

