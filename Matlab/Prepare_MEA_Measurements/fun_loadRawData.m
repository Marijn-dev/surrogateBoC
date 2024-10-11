function [t,dataR,chanIDR,I,I_stim,t_stim] = fun_loadRawData(data_file)
global pm

%Read raw data and required channel info
dataR = double(h5read(data_file,'/Data/Recording_0/AnalogStream/Stream_0/ChannelData'))/pm.loading.V_scale;
channelInfo = h5read(data_file,'/Data/Recording_0/AnalogStream/Stream_0/InfoChannel'); 
chanIDR = channelInfo.Label;

tmax = double(h5readatt(data_file,'/Data/Recording_0/','Duration'))./pm.loading.t_scale;
tmin = double(h5readatt(data_file,'/Data/Recording_0/','TimeStamp'))./pm.loading.t_scale;
t = linspace(tmin, tmax, size(dataR,1));

pm.filtersettings.sampling_frequency = size(dataR,1)/(tmax-tmin);
pm.st = 1/pm.filtersettings.sampling_frequency;

if isnan(pm.stim.channelID)
    I = zeros(size(dataR));
    I_stim = zeros(size(dataR,1),1);
    t_stim = [];
else
    %Load the event data
    eventInfo1 = h5read(pm.loading.measurement,'/Data/Recording_0/EventStream/Stream_0/EventEntity_1');
    startPulse_indx = int32((double(eventInfo1(:,1))/pm.loading.t_scale)*pm.filtersettings.sampling_frequency);
    eventInfo2 = h5read(pm.loading.measurement,'/Data/Recording_0/EventStream/Stream_0/EventEntity_2');
    stopPulse_indx = int32((double(eventInfo2(:,1))/pm.loading.t_scale)*pm.filtersettings.sampling_frequency);

    I = zeros(size(dataR));
    
    if strcmp(pm.stim.channelID,'Tunnels') 
        IDs = pm.stim.tunnelID;
    elseif strcmp(pm.stim.channelID,'reservoir A')
        IDs = pm.stim.resA_ID;
    elseif strcmp(pm.stim.channelID,'reservoir B')
        IDs = pm.stim.resB_ID;
    else
        IDs = pm.stim.channelID;
    end

    %Check if the channel is in the selected channels to be plotted
    for i = 1:size(IDs,1)
        j = find(strcmp(IDs(i,:), chanIDR));    
        I_stim = fun_GetI(size(dataR,1),startPulse_indx, stopPulse_indx);
        I(:,j) = I_stim;
    end
    
    t_stim = t(I_stim~=0);
    remove_t_stim = find(diff(t_stim)<2*pm.st)+1;
    % remove_t_stim = find(diff(t_stim)<0.1)+1;
    t_stim(remove_t_stim) = [];
        
end

disp('Raw data successfully loaded')

end

 