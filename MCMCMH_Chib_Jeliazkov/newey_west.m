function var_h = newey_west(h, m)
% Calculates robust to autocorrelation covariance matrices
% 
% Ramis Khabibullin (rawirtschaft@gmail.com)
    M = size(h, 1);
    mean_h = mean(h, 1);
    h = h - ones(size(h, 1), size(h, 2))*diag(mean_h);
    
    var_h = h'*h/M;
    for s = 1:m
        var_hs = h((s + 1):end, :)' * h((s + 1):end, :)/M;
        var_h  = var_h + (1 - s/(m + 1))*var_hs;
    end
    
    var_h = var_h/M;
end