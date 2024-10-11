function [dataR, I, t_sweep, ChIDs] = fun_loadSweepData(filename)
global pm

segment_info = h5read(filename, '/Data/Recording_0/SegmentStream/Stream_0/InfoSegment');
ChIDs_cells = flipud(segment_info.Label); %matching the dataR indices

%Extract raw data
ChIDs = strings(0);
for i=0:119
    dataR{i+1} = h5read(filename, strcat('/Data/Recording_0/SegmentStream/Stream_0/SegmentData_' ,int2str(i) ) )/pm.loading.V_scale;
    ChIDs(i+1) = ChIDs_cells{i+1};
end

%Extract time data
tpre = double(segment_info.PreInterval(1))*10^-3;
tpost = double(segment_info.PostInterval(1))*10^-3;
t_sweep = linspace(-tpre, tpost, size(dataR{1},2));
pm.filtersettings.sampling_frequency = size(dataR{1},2)/(tpre+tpost);

% %Filter the data  WORK OUT BETTER AND DO FILTERING PER SWEEP
% for i=0:119
%     dataF{i+1} = fun_FilterData(dataR{i+1},false); %Filter the data
% end

%Extract the stimulation
I = fun_GetI(t_sweep);

end

