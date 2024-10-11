close all; clc; clear all;
plot_settings
% fun_saveFig('spike_response')

global pm
save_traj = true;

%% Set up the parameters 
fun_parameters()
% fun_overview_h5file(pm.loading.measurement); fun_overview_print_h5file(pm.loading.measurement, 'test.txt')

%% Load in the data
[t,y,chanID,I,I_stim,t_stim] = fun_loadRawData(pm.loading.measurement);
% [y_sweep, I, t_sweep, ChIDs] = fun_loadSweepData(pm.loading.measurement);

%% Filter the data
y = fun_FilterData(y,false); %filter the data

%% Extract the spikes 
th = fun_FindThreshold(t,y);
[sp_struct, spk_times, spk_chan] = fun_getSpikes(t,y,th,t_stim);

%% Select the correct electrodes
chanIDselect = {'J5','J6','G5'};  
I = fun_SelectChanData(I,chanID,chanIDselect);

%% Calculate the firing rate
temp_res = 0.05;
window = 'Causal'; %Gaussian or Causal

% determine time of evaluation
% dt_eval = ((t(end)-t(1))/nt);
% t_eval = t(1):dt_eval:t(end);

[r,t_eval] = fun_calcFiringRate(chanIDselect,chanID,sp_struct,t,temp_res,-1,window,I_stim);

%% Plot trajectoreis
close all
fun_plotSelectChan(t_eval,r,chanIDselect,chanIDselect,[t_stim(1)-1 t_stim(1)+1],[0 30],t_stim) %[t(1) 20]

%% split data in equal traj of 10 sec.
% seq_t = 5;
% [data_t,data_fr,data_I] = fun_getTraj(t,r,I,seq_t,true);

% close all
% for i=1:length(data_t)
%     figure()
%     plot(data_t{i},data_fr{i}(:,1))
%     hold on
%     plot(data_t{i},data_I{i}(:,1))
% end

% if save_traj
%     %Save for the first time
%     mat_file = fullfile('Datasets', 'test') %['Exp_traj_' pm.n_MEA '_' pm.file_name '_.mat']);
%     save(mat_file, 'data_t', 'data_fr', 'data_I');
% end

%% Analyse timing on control horizon stim.duration increments of t_stim
len_t = length(t);
n_full_blocks = floor(len_t / 4);  % Number of full [1; 2; 3; 4] blocks
n_rem = rem(len_t, 4);             % Remaining elements

% Construct the full blocks
full_blocks = repmat((1:4)', n_full_blocks, 1);

% Construct the remaining elements
remaining_elements = (1:n_rem)';

% Combine the full blocks and remaining elements
delta = [full_blocks; remaining_elements];

stim_delta = [];
idx_t_stim = [];
for i = t_stim
    idx_t_stim = [idx_t_stim find(t==i)];
    stim_delta = [stim_delta delta(idx_t_stim(end))];
end


%% Calculate start and end indx, shift on delta of splits
% max_split_idx = 0.75/(1/pm.filtersettings.sampling_frequency); % Set the maximum time for each segment (e.g., 100000 units)
max_split_idx = 0.05/(1/pm.filtersettings.sampling_frequency); % Set the maximum time for each segment (e.g., 100000 units)
split_factor = 0.8; % Split factor (0.8% of the time between two stimulations)

% Initialize variables
end_idx = [];
start_idx = [max(idx_t_stim(1)-round(0.2*max_split_idx),1)];
I_time_idx = [];
not_yet_stim = 1:length(t_stim);

count_splits = 1;
for i = 1:length(t_stim)
    if ismember(i,not_yet_stim)
        fprintf('pulse %d \n',i)
        not_yet_stim(not_yet_stim==i) = [];
        
        fprintf('split %d with stim at: %d and start idx: %d \n',length(start_idx),idx_t_stim(i), start_idx(count_splits))

        end_idx = [end_idx start_idx(end)+max_split_idx];
        I_time_idx = [I_time_idx idx_t_stim(i)];

        for j = i+1:length(t_stim)
            if ~isempty(find(start_idx(end):end_idx(end)==idx_t_stim(j),1))
                               
                %Check if it is in the same window
                if stim_delta(i)==stim_delta(j)
                    %Another pulse with same time window in this split
                    not_yet_stim(not_yet_stim==j) = [];
                    fprintf('additional matching pulse in this traj, deleted pulse %d \n',j)
                else
                    %End split due to difference in window with other pulse
                    split_idx = idx_t_stim(i) + round(split_factor * (idx_t_stim(j)-idx_t_stim(i))); 
                    end_idx(end) = split_idx;

                    fprintf('split before pulse_idx: %d \n',end_idx(end))
                    % fprintf('the original time pulse was %d \n',idx_t_stim(i))
                    break
                end
            end
        end
        
        fprintf('The end idx: %d\n',end_idx(end))

        if i<length(t_stim)
            start_idx = [start_idx max(idx_t_stim(not_yet_stim(1))-round(0.2*max_split_idx),end_idx(end))];
        else
            end_idx(end) = min(length(t),end_idx(end));
        end
        count_splits = count_splits+1;
    end
end

%% Make trajectories with the start_idx, end_idx and the I_stim_idx
data_t = {}; % Output cell array to hold segments
data_fr = {};
data_I = {};
data_I_seq = {};
data_t_seq = {};


%calculate the shift
shift_delta = [];
for i=1:length(start_idx)
    Indx_stim_delta = fun_getStimDelta(start_idx(i):end_idx(i),I_time_idx(i));
    shift_delta(i) = Indx_stim_delta-1;
end

start_idx = start_idx + shift_delta;

% Save the current segment at the correct split
for i=1:length(start_idx)
    data_t{i} = t(1:end_idx(i)-start_idx(i)+1);
    data_fr{i} = r(start_idx(i):end_idx(i),:);
    data_I{i} = I(start_idx(i):end_idx(i),:);
    idx_seq = start_idx(i):(round(pm.stim.duration/pm.st)):end_idx(i);
    data_t_seq{i} = t(idx_seq-start_idx(i)+1);
    data_I_seq{i} = I(idx_seq,:);
end

if save_traj
    mat_file = fullfile('Datasets_splits_005', ['Exp_traj_' pm.n_MEA '_' pm.file_name '_.mat']);
    save(mat_file, 'data_t','data_fr','data_I','data_I_seq','data_t_seq');
end


%% Plot the splits
close all
for i=1:3 %length(data_t)
    figure()
    plot(data_t{i},data_fr{i}(:,1))
    hold on
    % plot(data_t{i},data_I{i}(:,1))
    plot(data_t{i},data_I{i}(:,1),'o')
    plot(data_t_seq{i}, data_I_seq{i}(:,1),'o')
    xlabel('time (s)')
    ylabel('Firing rate (Hz)')
end
