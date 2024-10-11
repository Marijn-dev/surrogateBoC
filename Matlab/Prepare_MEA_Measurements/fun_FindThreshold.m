function [th] = fun_FindThreshold(t,Data)
    global pm
    
    diff1 = abs(t-pm.baselinenoise.purenoise(1));
    diff2 = abs(t-pm.baselinenoise.purenoise(2));
    [~,ind1] = min(diff1);
    [~,ind2] = min(diff2);
    
    %Extract only pure noise from the data
    noise_data = {};
    for i=1:size(Data,2)
        noise_data{i} = Data(ind1:ind2,i);
    end
    
    %Calculate the RMS and threshold
    RMS={};
    th={};
    for i=1:length(noise_data)
        RMS{i} = sqrt(mean(noise_data{i}.^2));
        th{i} = pm.baselinenoise.RMS*RMS{i};
    end

    disp('Threshold for spike detection established')

end