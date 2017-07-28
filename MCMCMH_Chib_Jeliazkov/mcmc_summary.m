classdef mcmc_summary 
    properties
        series
        descr_stats
        quantiles
        which_q
        convergence
    end
    
    methods
        function obj = mcmc_summary(x, burn_in, L, use_newey_west)
            x = x(:, (burn_in + 1):end);
            obj.series = x;
            obj.descr_stats = nan(size(x, 1), 4);
            % Mean NWse sd skewness kurtosis
            obj.descr_stats(:, 1) = mean(x, 2); 
            if use_newey_west
                e = obj.series - obj.descr_stats(:, 1)* ...
                    ones(1, size(x, 2));
                for i = 1:size(x, 1) 
                    obj.descr_stats(i, 2) = NeweyWest(e(i, :)', ...
                        ones(size(x, 2), 1), L, 0);
                end
            end
            obj.descr_stats(:, 3) = sqrt(var(x, 0, 2));
            obj.descr_stats(:, 4) = skewness(x, 0, 2);
            obj.descr_stats(:, 5) = kurtosis(x, 0, 2);
            obj.which_q = [0.01, 0.025, 0.05, 0.1, 0.25, ...
                0.5, 0.75, 0.9, 0.95, 0.975, 0.99];
            obj.quantiles = quantile(x, obj.which_q, 2);
            % Convergence diagnostics
            obj.convergence.raftery = nan(1, 1);
        end
        
        function plot_mcmc_hist(obj, xbins, panel_size, benchmark)
            n_s = size(obj.series, 1);
            figure();
            j = 1;
            for i = 1:n_s
                subplot(panel_size(1), panel_size(2), j)
                
                [histFreq] = hist( obj.series(i, :), xbins);
                histFreq = histFreq / sum(histFreq);
                bar(xbins, histFreq);
                
                if (~isnan(benchmark(i)))
                     hold on
                     y1 = get(gca,'ylim');
                     plot([benchmark(i) benchmark(i)],y1, 'r')
                     hold off
                end
                
                if (rem(i, panel_size(1) * panel_size(2)) == 0)&&(i ~= n_s)
                    figure(); j = 1;
                else 
                    j = j + 1;
                end
            end
        end
    end
    
end