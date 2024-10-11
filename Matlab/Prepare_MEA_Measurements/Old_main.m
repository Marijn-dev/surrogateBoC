close all; clc; clear all;
plot_settings
% Code for saving an image: exportgraphics(gcf,'Figures/F6_Raw.emf','ContentType','vector','BackgroundColor','none')
% exportgraphics(gcf,'Figures/F6_Raw.eps','-depsc')

%% Variables
global pm
    
Measurement = 1; %Select 1 or 2
fun_parameters(Measurement)

%% Load the data
[t,dataR,chanIDR,~,~] = fun_loadData(pm.loading.datafile); %data in [uV]
fun_plotAll(t,dataR,chanIDR,[t(1) 0.2*t(end)],[-120 120])

%%
fun_plotAll(t,dataR,chanIDR)

%% Filter the data
dataF = fun_FilterData(dataR,false);
% exportgraphics(gcf,'Figures/filter.eps','depsc')
%% Extract selected channels
%Select a tunnel with the IDs
chanIDSelect = {'E6';'F6';'G6'}; 
% chanIDSelect = {'F6'}; 
[data_tun6F] = fun_SelectChanData(dataF,chanIDR,chanIDSelect);
fun_plotSelectChan(t,data_tun6F,chanIDSelect,[t(1) t(end)],[-1000 1000]) %-400 200 , -1000 1000

%% Stimulation reconstruction and Spike detection
[th,thStim] = fun_FindThreshold(t,data_tun6F,chanIDSelect,true,true);

[I6,I_stimCh,I_peaks,allPeaks] = fun_ConstructI(t,data_tun6F,chanIDSelect,thStim,th,true,[]);

[pk,t_pk] = fun_getSpikes(t,data_tun6F,chanIDSelect,th,I_peaks,true);

maxPk = cellfun(@min,pk);
NPk = cellfun(@length,pk);

%% Visualise the data
%Plot results
if pm.loading.measurement == 1
    fun_plotInputOutput(t,I_stimCh,data_tun6F,chanIDSelect,th,pk,t_pk, [I_peaks(1)-0.1 I_peaks(1)+1]) %[5 8] [9.9 11]
else
    fun_plotInputOutput(t,I_stimCh,data_tun6F,chanIDSelect,th,pk,t_pk,[])
end

%spike raster plot
fun_plotSpikeResponse(t_pk,chanIDSelect,I_peaks)

%% Analyse the time delays in the tunnel
% i=4; t_pk_Interval=[I_peaks(i) I_peaks(i+1)]; %mea1: 0-8, mea2: 0-5 10-15, 15-20
% t_pk_Interval = [I_peaks(1) I_peaks(1)+5]; %mea1: 0-8, mea2: 0-5 10-15, 15-20
t_pk_Interval = [I_peaks(1) 15];

refCH = chanIDSelect{2}; IgnCH = chanIDSelect{1}; 
% refCH = chanIDSelect{1}; IgnCH = []; 

[t_pk_srt_all] = fun_MatchTunnelSpikes(t,t_pk);
[delay,delayCL,sp_tot] = funTimeDelayTunnel(t_pk_srt_all,chanIDSelect,refCH,t_pk_Interval,IgnCH,[]);

