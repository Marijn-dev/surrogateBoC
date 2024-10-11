function [means,stds,medians,options] = fun_calc_mean(dataC,dataL)
means = [];
stds = [];
medians = [];

options = unique(dataC);
for i = 1:length(options)
    % Extract data for the current option
    currentData = dataL(dataC == options(i));
    
    % Calculate the IQR
    Q1 = quantile(currentData, 0.25);
    Q3 = quantile(currentData, 0.75);
    IQR = Q3 - Q1;
    
    % Define outliers as points outside 1.5 * IQR from Q1 and Q3
    lowerBound = Q1 - 1.5 * IQR;
    upperBound = Q3 + 1.5 * IQR;
    
    % Remove outliers
    nonOutliers = currentData(currentData >= lowerBound & currentData <= upperBound);
    
    % Calculate statistics without outliers
    avg_data = mean(nonOutliers);
    std_data = std(nonOutliers);
    med_data = median(nonOutliers);
    
    means = [means avg_data];
    stds = [stds std_data];
    medians = [medians med_data];
end
end
