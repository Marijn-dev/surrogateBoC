function [] = fun_parameters()
global pm

pm = [];

%% Parameters that can be changes:
pm.file_name = 'rec11';
pm.n_MEA = 'A';

if strcmp(pm.n_MEA,'A')
    pm.stim.tunnelID = ["J1";"K2";"J3";"J4";"J5";"J6";"J7";"J8";"J9";"K10";"K11";"J12"];
    pm.stim.resA_ID = ["K3","K5","K7","K9","L4","L6","L8","L10","M5","M6","M7","M9"]';
    pm.stim.resB_ID = ["A4","A6","A8","B3","B5","B7","B9","C4","C6","C8","C10"]';
elseif strcmp(pm.n_MEA,'B')
    pm.stim.tunnelID = ["J1";"H2";"G3";"G4";"G5";"G6";"G7";"G8";"G9";"H10";"J11";"J12"];
    pm.stim.resA_ID = ["H3","H5","H7","H9","J2","J4","J6","J8","J10","K3","K5","K7","K9","K11","L4","L6","L8","L10","M5","M7","M9"]';
    pm.stim.resB_ID = ["A5","A7","A9","B3","B6","B8","B10","C2","C4","C5","C7","C9","C10","D3","D6","D8","D10","E5","E7","E9"]';
end

[StimID, amp, dur] = fun_getStimProp(pm.n_MEA, [pm.file_name '.h5']);
pm.stim.channelID = StimID;
pm.stim.amplitude = amp; %[mV / muA]
pm.stim.duration = dur; %[s]

pm.loading.t_scale = 10^6; %[s]
pm.loading.V_scale = 16.7771; %Scaling the axis [muV]
pm.loading.measurement = ['D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\MEA_Recordings\0207_data_Guillem\' pm.n_MEA '\' pm.file_name '.h5'];

pm.save_results = true;

test_name = 'rec11_tstim1_short_horizon';
pm.figure_folder = fullfile('D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\Matlab_ExpData_with_EKF_FHN\', ['Figures_thesis\' test_name]);
if ~exist(pm.figure_folder, 'dir') && pm.save_results
   mkdir(pm.figure_folder)
end

pm.standardsettings.minimum_activity_channel = 0.1;
pm.standardsettings.sizeOfMatrix = [12 12];
pm.standardsettings.column_strings = ['A'; 'B'; 'C'; 'D'; 'E'; 'F'; 'G'; 'H'; 'J'; 'K';'L';'M'];

pm.filtersettings.cut_off_frequency=200;
pm.filtersettings.filter_order=2;

pm.baselinenoise.purenoise=[0 2]; %The interval in s at of the measurement where there is only noise
pm.baselinenoise.window=50; %represents the size of each bin when the data is split in smaller windows to determine which section of the recording is pure noise
pm.baselinenoise.purenoisewindow=2; %indicates the minimum duration of the time window that is used to establish the baseline noise.
pm.baselinenoise.RMS=5; %represents how strict the threshold is for spike detection. The higher this number is, the stricter the threshold

pm.spike.minpeakdistance=0.02; %[s] minimum distance between 2 peaks
pm.spike.minpeakprominence=0; %Another peak detection feature
pm.spike.maxdelayinTunnel=0.001; %[s] the maxmimum delay between 2 electrodes in a tunnel
    
pm.burstdetection.ISI=0.1;
pm.burstdetection.nspikes=10;
pm.burstdetectionmaxinterval.start=0.17;        %used in the paper 0.05  
pm.burstdetectionmaxinterval.nspikes=10;        %used in the paper 4
pm.burstdetectionmaxinterval.IBI=0.3;           %used in the paper 0.1
pm.burstdetectionmaxinterval.intraBI=0.2;       %used in the paper 0.1
pm.burstdetectionmaxinterval.mindur=0.01;       %used in the paper 0.03
pm.burstdetectionlogisi.void=0.7;
pm.burstdetectionlogisi.nspikes=3;
pm.networkburstdetection.synchronizedtimewindow = 0.1; % time in seconds
pm.networkburstdetection.minimumsynchronizedburstcount = 2; % in counts
pm.networkburstdetection.minimumchannelparticipation = 0.5; % in percentage

disp(['Simulation is using file: ', pm.file_name, ', MEA ', pm.n_MEA]);
disp(['Stimulation channel: ', StimID]);
end

