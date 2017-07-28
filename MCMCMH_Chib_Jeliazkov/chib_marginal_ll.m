function [log_ml, var_ml] = chib_marginal_ll(A0, params, prior_sd, Y, Z, isConstrained, calc_var, nBurn, nDraw)
% Calculates marginal likelihood by Chib, Jeliazkov method
% 
% Ramis Khabibullin (rawirtschaft@gmail.com)
    T = size(Y, 1);
    [m, n] = size(A0);
    var_ml = 0;
    
    indx =  reshape(1:(m*n), m, n);
    
    D0 = diag(A0).^2;
    A0 = A0 / diag(sqrt(D0));
    A0(isConstrained) = 0;
    sdProp = 0.15;
    
    post_pars = define_post_params(Y, Z, m, isConstrained);
    
    % 1. Sum log posterior density p(B, D| A, X)
    log_post_BD = log_post_B_dens(A0, D0, post_pars, isConstrained);
    log_post_BD = log_post_BD + log_post_D_dens(A0, D0, post_pars, T, n);
%     log_post_BD = log_post_B_dens(A0, D0, post_pars) + ...
%         log_post_D_dens(A0, D0, post_pars, T, n);
    
    % 2. Calculate log posterior density p(A | B, D, X)
    if (sum(sum((~isConstrained(1:n, :)))) == n + 1)
        % 2a. One MH block case
        % 2a.1 Sample from conditional densities
        which_sample = [diag(ones(n, 1)); ones(m - n, n)];
        
        [Ahat, ~, ~] =  ...
            mcmc_bad2(isConstrained, A0 * diag(sqrt(D0)), which_sample,  nBurn, nDraw);
        % 2a.2 Sample from proposal density and calculate acceptance ratios
        % that should be summed in the denominator
        cond = (which_sample == 0)&(~isConstrained);
        j = indx(cond);
        
        AR_denom = zeros(size(Ahat, 1), 1);
        for i_rep = 1:size(Ahat, 1)
            Aold = reshape(Ahat(i_rep, :), m , n);
            Dold = diag(Aold).^2; Aold = Aold / diag(sqrt(Dold));
            Aprop = Aold;
            Aprop(j) = A0(j) + randn(1) * sdProp;
            AR_denom(i_rep) = ...
                    exp(nll_bad(Aold, Dold) - nll_bad(Aprop, Dold));
            AR_denom(i_rep) = (AR_denom(i_rep) > 1) * 1 +  ...
                (AR_denom(i_rep) <= 1)*AR_denom(i_rep); 
        end
        % 2a.3 Calculate acceptance ratios for MH output
        fun = zeros(size(params, 1), 1);
        for i_rep = 1:size(params, 1)
             Aold = reshape(params(i_rep, :), m , n);
             Dold = diag(Aold).^2; Aold = Aold / diag(sqrt(Dold));
             Aprop = Aold;
             Aprop(j) = A0(j);
             fun(i_rep) = ...
                    exp(nll_bad(Aold, Dold) - nll_bad(Aprop, Dold));
             fun(i_rep) = (fun(i_rep) > 1) * 1 +  ...
                (fun(i_rep) <= 1)*fun(i_rep);
             fun(i_rep) = fun(i_rep)*normpdf(Aprop(j), Aold(j), sdProp);
        end
        log_post_A = log(mean(fun)/mean(AR_denom));
        
        if calc_var
            % A1. Calculate density of reduced run parameters
            cond_dens = ones(size(Ahat, 1), 1);
            for i_rep = 1:size(Ahat, 1)
                As = reshape(Ahat(i_rep, :), m , n);
                Ds = diag(As).^2; As = As / diag(sqrt(Ds));
                cond_dens(i_rep) = log_post_B_dens(As, Ds, post_pars, ...
                    isConstrained) + log_post_D_dens(A0, D0, post_pars, T, n);
            end
            % A2. Calculate the h matrix
            h = [fun, AR_denom, cond_dens];
            var_h = newey_west(h, 1);
            var_ml = [1, -1, 1] * var_h * [1; -1; 1];
        end
    else
        % 2b. Two MH block case
        % 2b.1 Sample from conditional densities B, D, a_{12} | a_{21}
         which_sample = [diag(ones(n, 1)); ones(m - n, n)];
         which_sample(1, 2) = 1;
        
        [Ahat1, ~, ~] =  ...
            mcmc_bad2(isConstrained, A0 * diag(sqrt(D0)), which_sample,  nBurn, nDraw);
        % 2b.2 Calculate acceptance ratios for  the last step output
        j = indx(1, 2);
        
        fun1 = zeros(size(Ahat1, 1), 1);
        for i_rep = 1:size(Ahat1, 1)
             Aold = reshape(Ahat1(i_rep, :), m , n);
             Dold = diag(Aold).^2; Aold = Aold / diag(sqrt(Dold));
             Aprop = Aold;
             Aprop(j) = A0(j);
             fun1(i_rep) = ...
                    exp(nll_bad(Aold, Dold) - nll_bad(Aprop, Dold));
             fun1(i_rep) = (fun1(i_rep) > 1) * 1 +  ...
                (fun1(i_rep) <= 1)*fun1(i_rep);
             fun1(i_rep) = fun1(i_rep)*normpdf(Aprop(j), Aold(j), sdProp);
        end
        
        % 2b.3 Sample B, D | a_{21}, a_{12}
        which_sample = [diag(ones(n, 1)); ones(m - n, n)];
        [Ahat2, ~, ~] =  ...
            mcmc_bad2(isConstrained, A0 * diag(sqrt(D0)), which_sample,  nBurn, nDraw);
        
        % 2b.4 Calculate acceptance ratios for denominator for a_{12}
        AR_denom1 = zeros(size(Ahat2, 1), 1);
        for i_rep = 1:size(Ahat2, 1)
            Aold = reshape(Ahat2(i_rep, :), m , n);
            Dold = diag(Aold).^2; Aold = Aold / diag(sqrt(Dold));
            Aprop = Aold;
            Aprop(j) = A0(j) + randn(1) * sdProp;
            AR_denom1(i_rep) = ...
                    exp(nll_bad(Aold, Dold) - nll_bad(Aprop, Dold));
            AR_denom1(i_rep) = (AR_denom1(i_rep) > 1) * 1 +  ...
                (AR_denom1(i_rep) <= 1)*AR_denom1(i_rep); 
        end
        log_post_A = log(mean(fun1)/mean(AR_denom1));
        
        % 2b.5 Calculate acceptance rates for a_{21}
        j = indx(2, 1);
        
        AR_denom2 = zeros(size(Ahat1, 1), 1);
        for i_rep = 1:size(Ahat1, 1)
            Aold = reshape(Ahat1(i_rep, :), m , n);
            Dold = diag(Aold).^2; Aold = Aold / diag(sqrt(Dold));
            Aprop = Aold;
            Aprop(j) = A0(j) + randn(1) * sdProp;
            AR_denom2(i_rep) = ...
                    exp(nll_bad(Aold, Dold) - nll_bad(Aprop, Dold));
            AR_denom2(i_rep) = (AR_denom2(i_rep) > 1) * 1 +  ...
                (AR_denom2(i_rep) <= 1)*AR_denom2(i_rep); 
        end
        
        % 2b.6 Calculate numenator
        fun2 = zeros(size(params, 1), 1);
        for i_rep = 1:size(params, 1)
             Aold = reshape(params(i_rep, :), m , n);
             Dold = diag(Aold).^2; Aold = Aold / diag(sqrt(Dold));
             Aprop = Aold;
             Aprop(j) = A0(j);
             fun2(i_rep) = ...
                    exp(nll_bad(Aold, Dold) - nll_bad(Aprop, Dold));
             fun2(i_rep) = (fun2(i_rep) > 1) * 1 +  ...
                (fun2(i_rep) <= 1)*fun2(i_rep);
             fun2(i_rep) = fun2(i_rep)*normpdf(Aprop(j), Aprop(j), sdProp);
        end
        log_post_A = log_post_A + log(mean(fun2)/mean(AR_denom2));
        
        if calc_var
            % A1. Calculate density of reduced run parameters
            cond_dens = ones(size(Ahat2, 1), 1);
            for i_rep = 1:size(Ahat2, 1)
                As = reshape(Ahat2(i_rep, :), m , n);
                Ds = diag(As).^2; As = As / diag(sqrt(Ds));
                cond_dens(i_rep) = log_post_B_dens(As, Ds, post_pars, ...
                    isConstrained) + log_post_D_dens(A0, D0, post_pars, T, n);
            end
            % A2. Calculate the h matrix
            h = [fun1, AR_denom1, fun2, AR_denom2, cond_dens];
            var_h = newey_west(h, 1);
            var_ml = [1, -1, 1, -1, 1] * var_h * [1; -1; 1; -1; 1];
        end
    end
    
    
    % 3. Calculate prior densities
    A0 = A0 * diag(sqrt(D0));
    prior_dens =  sum(log(normpdf(A0(isConstrained), 0, prior_sd(1))));
    prior_dens =  prior_dens + sum(log(normpdf(A0(~isConstrained), ...
        0, prior_sd(2))));
    
    log_ml = -nll_bad(A0, D0) + prior_dens - log_post_BD - log_post_A;
end

function res = log_post_D_dens(A, D, post_pars, T, n) 
    res = 0;
    for j = 1:n
        tau_j = 0.5 * T * A(1:n, j)' * post_pars{j}.Omega * A(1:n, j);
        res = res + log(gampdf(D(j)^2, T/2, 1/tau_j));
    end
end

function res = log_post_B_dens(A, D, post_pars, isConstrained)
    [m, n] = size(A);
    res = 0;
    for j = 1:n
        B_var  = post_pars{j}.Bvar./D(j);
        B_mean = post_pars{j}.Phimean* A(1:n, j);
        
        if sum(size(B_var)) ~= 0
            A_temp = A((n+1):m, j); 
            res = res + log(mvnpdf(A_temp(~isConstrained((n+1):m, j)),...
                B_mean, B_var));
        else 
            res = res + 0;
        end
    end
end

function res = define_post_params(Y, Z, m,  isConstrained)
    [T, n] = size(Y); 
    res = cell(1, n);
    YY = Y' * Y;
    for j = 1:n
        restr = isConstrained((n + 1):m, j) == 0;
        ZZ = Z(:, restr)' * Z(:, restr);
        ZY = Z(:, restr)' * Y;
        res{j}.Bvar = inv(ZZ);
        res{j}.Phimean =  - ZZ \ ZY;
        
        res{j}.Omega = (YY - ZY' * res{j}.Bvar * ZY)/T;
    end
end

