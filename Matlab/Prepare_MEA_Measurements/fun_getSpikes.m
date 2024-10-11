function [all_spk, all_spike_times, all_channel_indices] = fun_getSpikes(t,y,th,t_stim)
global pm

warning('off','all')
%% Extract positive and negative spikes
all_spk = cell(size(y,2),2);

for i=1:size(y,2)
    [pk_pos,tk_pos] = findpeaks(y(:,i),t,'MinPeakHeight',th{i},'MinPeakDistance',pm.spike.minpeakdistance);
    [pk_neg,tk_neg] = findpeaks(-y(:,i),t,'MinPeakHeight',th{i},'MinPeakDistance',pm.spike.minpeakdistance);
    
    tk_tot = [tk_pos, tk_neg];
    pk_tot = [pk_pos; -pk_neg];
    
    [tk_tot_sort,ind_c,~] = unique(tk_tot);
    pk_tot_sort = pk_tot(ind_c);

    all_spk{i,1} = tk_tot_sort;
    all_spk{i,2} = pk_tot_sort;

    %% Remove artifacts: 
    %https://authors.library.caltech.edu/25142/1/01419673.pdf
    % First Criteria is that the peak found needs to be the highest in
    % ampltiude and within a 1 ms window there can't be any secondary
    % peaks in the same polarity
    % the second criteria is that 50% of the ampltiude of the peak
    % needs to be in the same window?
    % the third criteria is that spikes with the exact same time stamps
    % on 2 or more channels are not considered to be spikes

    toberemoved = find(diff(all_spk{i,1})<0.001)+1;
    if isempty(toberemoved)
    else
        % now we need to check which peak has the highest amplitude and
        % remove the one that is lower
        test = all_spk{i,2}(toberemoved);
        test2 = all_spk{i,2}(toberemoved-1);
        test3 = test > test2 ; % 0 means that the number in test2 is higher and 1 means that the number in test is higher
        for ii = 1:length(test3)
            if test3(ii) == 1
                toberemoved(ii) = toberemoved(ii)-1;
            end
        end
        
        all_spk{i,1}(toberemoved) = [];
        all_spk{i,2}(toberemoved) = [];
    end
    
    %% Remove artifacts based on stim events
    tol = 0.002;
    for j=1:length(t_stim)
        % toberemoved = abs(all_spk{i,1}-t_stim(j))<0.002;
        toberemoved = t_stim(j)-all_spk{i,1}-tol<0 & all_spk{i,1}<t_stim(j)+pm.stim.duration+tol;

        all_spk{i,1}(toberemoved) = [];
        all_spk{i,2}(toberemoved) = [];
    end

    %% Empty channels without a minimum activity of 0.1 Hz
    N = length(all_spk{i,1});
    if (N/t(end))< pm.standardsettings.minimum_activity_channel
        all_spk{i,1} = [];
        all_spk{i,2} = [];
    end
end

%% Additional adjustment for easy plotting
all_spike_times = [];
all_channel_indices = [];
for i=1:size(y,2)
    all_spike_times = [all_spike_times all_spk{i,1}];
    all_channel_indices = [all_channel_indices i*ones(size(all_spk{i,1}))];    
end

disp('Spikes detected')

warning('on','all')


end