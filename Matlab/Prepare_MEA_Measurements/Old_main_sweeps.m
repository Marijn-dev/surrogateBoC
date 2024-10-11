close all; clc; clear all;
global pm

fun_parameters()
plot_settings

%% Get the info of the data
% data_file_sweeps = 'D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\MEA_Recordings\Data_06_18\DeviceB\sweeps\2024-06-18T10-43-36McsRecording.h5';
data_file_sweeps = 'D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\MEA_Recordings\0207_data_Guillem\A\rec11.h5';
% fun_overview_print_h5file(data_file_sweeps,'DataOverview_sweepANDraw_data.txt')

%% Load the data
[dataR, I, t_sweep, ChIDs] = fun_loadSweepData(data_file_sweeps);

%% Plot a test data
% Checkout = ["J3" "H3" "G3" "F3"];
% Checkout = ["H4" "G4" "F4"];
% Checkout = ["G5" "F5" "E5"];
% Checkout = ["G6" "F6" "E6"];
% Checkout = ["G7" "F7" "E7"];
% Checkout = ["G8" "F8" "E8"];
% Checkout = ["G9" "F9" "E9"]; 
% Checkout = ["G10" "F10" "E10"]; 
% Checkout = ["G11" "F11" "E11"]; 
Checkout = ["J5" "H5" "G5" "F5" "E5" "D5"];

[~, ind] = ismember(Checkout,ChIDs);

l=1;
for k = ind
    ind_data{l} = dataR{k};
    l=l+1;
    test_data = dataR{k};

%     figure()
%     for j=1:10
%         subplot(5,2,j)
%         plot(t_sweep,test_data(j,:))
%         hold on
%         ylabel('Potential ($\mu V$)')
%         plot(t_sweep,I,'r')
%         ylabel('V ($\mu V$)')
%         xlim([-0.05 0.08])
%     %     xlim([-tpre tpost])
%         ylim([-300 300])
%         title(strcat('Stim ',string(j),' on ',ChIDs(k)))
%     end
%     xlabel('time (s)')
%     legend('Measured data','Stimulation')
%     subplot(5,2,j-1),,
%       xlabel('time (s)')
end
%% 

colours = [	"#0072BD" "#D95319" "#7E2F8E" "#77AC30" "#EDB120" "#4DBEEE" "#A2142F"];

num_stim = 2;
figure()
for j= 1:num_stim
    subplot(num_stim,1,j)
    plot(t_sweep,I,'g')
    hold on
    for k=1:length(Checkout)
        dataind = ind_data{k};
        plot(t_sweep,dataind(j,:),'Color',colours(k))
        hold on
        xlabel('Time (s)')
        ylabel('Potential ($\mu V$)')
        xlim([-0.01 1])
        ylim([-500 500])
%         xlim([-0.05 0.5])
%         ylim([-1.05*pm.stim.amplitude 1.05*pm.stim.amplitude])
    end
end
legend(["stim in J5" Checkout])

%%
% figure()
% plot(t_sweep,test_data(j,:))
% hold on
% ylabel('Membrane potential ($\mu V$)')
% plot(t_sweep,I,'r')
% ylabel('V ($\mu V$)')
% xlim([0.1*t_sweep(1) 0.05])
% %     xlim([-tpre tpost])
% ylim([-500 500])
% title(strcat('Stim ',' ',string(j),' on ',ChIDs(k)))
