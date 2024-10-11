close all; clc; clear all;
plot_settings

save_plots = false;

%% Load results
exp_name = 'PI_impact_trainingParam/Sim_13/';
file_name = 'LF_PITrue_Nr3_Nu1_Nseq50';
arch = 'LF';
% 
% exp_name = 'PI_impact_trainingParam/Sim_13/';
% file_name = 'DON_PITrue_Nr3_Nu1_Nseq50';
% arch = 'DON';

data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data = readtable(data_name);
data_modelPar = load(data_modelPar_name);

%delete data.eta_gain~="0.001"
% data = data(data.eta_gain~=0.001,:);
% data = data(data.eta_gain~=0.001,:);

data.eta_gain = string(data.eta_gain);
data.eta_gain(data.phys_gain==0) = 'No Phys';
data.L_test_tot = data.L_test_NN + data.L_test_Phys;
data_with_phys = data(data.phys_gain~=0,:);

figure_folder = fullfile(['Results_sim/' exp_name], ['Figures_' arch]);
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

%% Plotting
loss_type_plot = 'L_test_Phys';

figure()
plt = groupedSpacedBoxchart(data_with_phys,'eta_gain',loss_type_plot,'init_gain');
hleg = legend(categorical(unique(data.init_gain)),'Location', 'northwest');
set(gca, 'YScale', 'log');
title(hleg,'Initial $\lambda$')
set(hleg, 'FontSize', 12);
xlabel('Learning rate $\eta_{\lambda}$');
ylabel('Test loss physics term (-)');
% title('Impact of physics gain hyperparameters')
ylim([2*10^-3 0.5])

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, [arch '_PI_impact_phys_loss.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder, [arch '_PI_impact_phys_loss.png']));
end

loss_type_plot = 'L_test_NN';

figure()
plt = groupedSpacedBoxchart(data,'eta_gain',loss_type_plot,'init_gain');
hleg = legend(categorical(unique(data.init_gain)),'Location', 'southeast');
set(gca, 'YScale', 'log');
title(hleg,'Initial $\lambda$')
set(hleg, 'FontSize', 12);
xlabel('Learning rate $\eta_{\lambda}$');
ylabel('Test loss data term (-)');
% title('Impact of physics gain hyperparameters')
ylim([4*10^-3 1])

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, [arch '_PI_impact_data_loss.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder, [arch '_PI_impact_data_loss.png']));
end

%% Extract optimal settings
c_filter = data.eta_gain=="0.005" & data.init_gain==0.1;
loss_testNN_optim = data.L_test_NN(c_filter);
loss_testPhys_optim = data.L_test_Phys(c_filter);
gain_optim = data.phys_gain(c_filter);

% Calculate mean and standard deviation
mean_loss_testNN = mean(loss_testNN_optim);
std_loss_testNN = std(loss_testNN_optim);

mean_loss_testPhys = mean(loss_testPhys_optim);
std_loss_testPhys = std(loss_testPhys_optim);

mean_gain = mean(gain_optim);
std_gain = std(gain_optim);

% Display the results with labels
fprintf('Mean of L_test_NN: %.4f\n', mean_loss_testNN);
fprintf('Standard deviation of L_test_NN: %.4f\n', std_loss_testNN);

fprintf('Mean of L_test_Phys: %.4f\n', mean_loss_testPhys);
fprintf('Standard deviation of L_test_Phys: %.4f\n', std_loss_testPhys);

fprintf('Mean of phys_gain: %.4f\n', mean_gain);
fprintf('Standard deviation of phys_gain: %.4f\n', std_gain);
%% Try overlap plot
% % Extract the default color order
% defaultColors = get(gca, 'ColorOrder');
% customColors = defaultColors([1, 2, 3, 6, 4, 5], :);
% 
% figure()
% loss_type_plot = 'L_test_NN';
% plt = groupedSpacedBoxchart(data,'eta_gain',loss_type_plot,'init_gain');
% hold on 
% loss_type_plot = 'L_test_Phys';
% plt = groupedSpacedBoxchart(data_with_phys,'eta_gain',loss_type_plot,'init_gain');
% legendLabels = [string(unique(data_with_phys.init_gain)) + ", Data"; string(unique(data.init_gain))+ ", Phys"];
% hleg = legend(legendLabels,'Location', 'best');
% title(hleg,'$\lambda_0$, $\mathcal{L}$-term')
% set(gca, 'YScale', 'log');
% set(hleg, 'FontSize', 10);
% xlabel('Learning rate $\eta_{\lambda}$');
% ylabel('Test loss value (-)');
% % title('Impact of physics gain hyperparameters')
% set(gca, 'ColorOrder', customColors, 'NextPlot', 'replacechildren');
% ylim([1*10^-3 4*10^0])
% 
% % Save the figure
% if save_plots
%     saveas(gcf, fullfile(figure_folder, [arch '_PI_impact_combine.eps']), 'epsc');
%     saveas(gcf, fullfile(figure_folder, [arch '_PI_impact_combine.png']));
% end
