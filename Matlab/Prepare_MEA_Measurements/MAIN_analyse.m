close all; clc; clear all;
plot_settings
% fun_saveFig('spike_response')

global pm

%% Set up the parameters 
fun_parameters()
% fun_overview_h5file(pm.loading.measurement); fun_overview_print_h5file(pm.loading.measurement, 'test.txt')

%% Load in the data
[t,y,chanID,I,I_stim,t_stim] = fun_loadRawData(pm.loading.measurement);
% [y_sweep, I, t_sweep, ChIDs] = fun_loadSweepData(pm.loading.measurement);

%% Filter the data
y = fun_FilterData(y,false); %filter the data
th = fun_FindThreshold(t,y);
% fun_plotAll(t,y,t_stim,chanID,th,false,[0 10],[])
% fun_saveFig('Thesis_figure_raw_data')

%% Extract the spikes 
[sp_struct, spk_times, spk_chan] = fun_getSpikes(t,y,th,t_stim);
fun_plotSpikeTrain(sp_struct,chanID,chanID,[0 10],[],t_stim,true)

% % plotID = {'J5';'J4';'J6'};
% fun_plotSpikeTrain(sp_struct,chanID,plotID,[],[],t_stim,false)

%% Calculate the firing rate
chanIDselect = chanID;  

% determine time of evaluation
dt_eval = 0.01; %((t(end)-t(1))/nt);
t_eval = t(1):dt_eval:t(end);
% chanIDselect,chanID,sp_struct,t,nt,sigma_w,num_closest
temp_res = 0.1;
window = 'Causal'; %Gaussian or Causal
[r,t_eval] = fun_calcFiringRate(chanIDselect,chanID,sp_struct,t_eval,temp_res,-1,window,I_stim);

%%
fun_plotAll(t_eval,r,t_stim,chanID,th,false,[0 10],[0 50])
fun_saveFig('All_FiringRate')


%% Plot all the channels
% Plot all
% close all
% fun_plotAll(t,y,t_stim,chanID,th,false,[0 20],[])
% fun_saveFig('All_Volt')
% fun_plotAll(t_eval,r,t_stim,chanID,th,false,[0 20],[0 50])
% fun_saveFig('All_FiringRate')

% close all
% 
% fun_plotSpikeTrain(sp_struct,chanID,chanID,[0 10],[],t_stim,true)
% fun_saveFig('All_spikeTrain')

%% Plot random channels indicating the th
% fun_plotThRandomChannels(y, t, th, chanID, 4)
% fun_saveFig('All_rndTh')

%% Plot tunnel that is stimulated
% plotID = {'J5';'H5';'G5';'F5';'E5'};
% % plotID = {'J6';'H6';'G6';'F6';'E6'};
% fun_plotSelectChan(t_eval,r,chanID,plotID,[],[],t_stim)
% fun_saveFig('Tun_FiringRate')
% fun_plotSelectChan(t,y,chanID,plotID,[],[-300 300],t_stim)
% fun_saveFig('Tun_Volt')
% fun_plotSpikeTrain(sp_struct,chanID,plotID,[],[],t_stim,false)
% fun_saveFig('Tun_SpikeTrain')
% 
% % Plot other active electrodes 
% % plotID = {'J5';'J4';'J6'};
% close all
% plotID = {'J4';'K4';'J5'; 'K5';'J6';'K6'};
% fun_plotSelectChan(t_eval,r,chanID,plotID,[],[0 45],t_stim)
% % fun_saveFig('ResA_FiringRate')
% % fun_plotSelectChan(t,y,chanID,plotID,[],[-300 300],t_stim)
% % fun_saveFig('ResA_Volt')
% fun_plotSpikeTrain(sp_struct,chanID,plotID,[],[],t_stim,false)
% % fun_saveFig('ResA_SpikeTrain')

%% Plot all tunnels midway through
% close all
% plotID = {'G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12'};
% fun_plotSelectChan(t_eval,r,chanID,plotID,[t(1) 10],[0 20],t_stim)
% fun_plotSelectChan(t,y,chanID,plotID,[t(1) 5],[-300 300],t_stim)




