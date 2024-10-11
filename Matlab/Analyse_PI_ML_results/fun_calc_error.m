function data = fun_calc_error(data)
    for i=1:length(data.est_alpha)
        x = data.est_alpha(i);
        y = data.true_alpha(i);
        y_norm = y{1};
        y_norm(y_norm==0) = 1;
        % data.error_alpha(i) = sqrt(mean((x{1} - y{1}).^2)); 
        data.error_alpha(i) = sqrt(mean(((x{1} - y{1})./y_norm).^2)); %mean((x{1}-y{1}).^2);
        
        x = data.est_beta(i);
        y = data.true_beta(i);
        y_norm = y{1};
        y_norm(y_norm==0) = 1;
        data.error_beta(i) = sqrt(mean(((x{1} - y{1})./y_norm).^2));
    
        x = data.est_gamma(i);
        y = data.true_gamma(i);
        data.error_gamma(i) = sqrt(mean(((x{1} - y{1})./y_norm).^2)); %.^2));
    
        x = data.est_W(i);
        y = data.true_W(i);
        y_norm = y{1};
        y_norm(y_norm==0) = 1;
        data.error_W(i) = sqrt(mean(((x{1} - y{1})./y_norm).^2));
    end
end