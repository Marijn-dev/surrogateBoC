close all; clc; clear all;
plot_settings

save_plots = true;

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,2,6,7,3,4,5], :);
close all

%% Load results
% Load the LF data
exp_name = 'Noise_disturbance/Sim_13/';
file_name = 'LF_PITrue_Nr3_Nu1_Nseq50';
data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data_LF = readtable(data_name);
data_modelPar_LF = load(data_modelPar_name);
data_LF.L_test_tot = data_LF.L_test_NN + data_LF.phys_gain.*data_LF.L_test_Phys;

figure_folder = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

% Load the DON data
exp_name = 'Noise_disturbance/Sim_13/';
file_name = 'DON_PITrue_Nr3_Nu1_Nseq50';
data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data_DON = readtable(data_name);
data_modelPar_DON = load(data_modelPar_name);
data_DON.L_test_tot = data_DON.L_test_NN + data_DON.phys_gain.*data_DON.L_test_Phys;

figure_folder = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

scale = [3*10^-3 20];

%% Organise the data to plot
x = length(data_DON.L_test_NN);
std_noise = [data_DON.noise_sigma; data_LF.noise_sigma];
data_Loss = [data_DON.L_test_NN; data_LF.L_test_NN];
phys_Loss = [data_DON.L_test_Phys; data_LF.L_test_Phys];
tot_Loss = [data_DON.L_test_tot; data_LF.L_test_tot];
architect = [repmat("DON", x,1); repmat("LF", x,1)]; 

table_combined_result = table(architect,std_noise,data_Loss,phys_Loss,tot_Loss);

%% Plot the combined plots
figure()
plt = groupedSpacedBoxchart(table_combined_result,'std_noise','data_Loss','architect');
hold on 
plt = groupedSpacedBoxchart(table_combined_result,'std_noise','phys_Loss','architect');
legendLabels = ["Data, " + string(unique(table_combined_result.architect)); "Phys, " + string(unique(table_combined_result.architect))];
hleg = legend(legendLabels,'Location', 'northwest');
title(hleg,'$\mathcal{L}$-term, Architect')
% set(hleg, 'FontSize', 10);
ylim([scale])
% title('Upscaling results')
set(hleg, 'FontSize', 12);
set(gca, 'YScale', 'log');
xlabel('Standard deviation noise');
ylabel('Test Loss value (-)');
set(gca, 'ColorOrder', customColors, 'NextPlot', 'replacechildren');
xtickangle(0)

if save_plots
    saveas(gcf, fullfile(figure_folder, 'box_noise.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'box_noise.png'));
end

%% Old analysis
[means_LF_data,stds_LF_data,medians_LF_data,options_LF_data] = fun_calc_mean(data_LF.noise_sigma,data_LF.L_test_NN);
[means_LF_phys,stds_LF_phys,medians_LF_phys,options_LF_phys] = fun_calc_mean(data_LF.noise_sigma,data_LF.L_test_Phys);
[means_DON_data,stds_DON_data,medians_DON_data,options_DON_data] = fun_calc_mean(data_DON.noise_sigma,data_DON.L_test_NN);
[means_DON_phys,stds_DON_phys,medians_DON_phys,options_DON_phys] = fun_calc_mean(data_DON.noise_sigma,data_DON.L_test_Phys);

opt = options_LF_data;

figure()
plot(opt, means_LF_data,'Color',customColors(1,:));
hold on
plot(opt, means_LF_phys,'Color',customColors(2,:));
plot(opt, means_DON_data,'Color',customColors(3,:));
plot(opt, means_DON_phys,'Color',customColors(4,:));

fill([opt', fliplr(opt')], [means_LF_data'+stds_LF_data'; flipud(means_LF_data'-stds_LF_data')],customColors(1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill([opt', fliplr(opt')], [means_LF_phys'+stds_LF_phys'; flipud(means_LF_phys'-stds_LF_phys')],customColors(2,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill([opt', fliplr(opt')], [means_DON_data'+stds_DON_data'; flipud(means_DON_data'-stds_DON_data')],customColors(3,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill([opt', fliplr(opt')], [means_DON_phys'+stds_DON_phys'; flipud(means_DON_phys'-stds_DON_phys')],customColors(4,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
legendLabels = ["Data, " + string(unique(table_combined_result.architect)); "Phys, " + string(unique(table_combined_result.architect))];
hleg = legend(legendLabels,'Location', 'northwest');
title(hleg,'$\mathcal{L}$-term, Architect')
set(hleg, 'FontSize', 12);
set(gca, 'YScale', 'log');
xlabel('Standard deviation noise');
ylabel('Test Loss value (-)');

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, 'means_noise.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'means_noise.png'));
end


%%

% figure();
% % plot(options, means);
% plot(options, medians);
% hold on
% fill([options', fliplr(options')], [means'+stds'; flipud(means'-stds')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% % fill([options', fliplr(options')], [medians'+stds'; flipud(medians'-stds')], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% xlabel('Standard deviation noise');
% ylabel('Test loss (-)');
% % title('Robustness to noise')
% % set(gca, 'YScale', 'log');
% % ylim(scale)



%%
% noise_c = categorical(data_LF.noise_sigma, unique(data_LF.noise_sigma));
% 
% figure()
% plt = boxchart(noise_c,plot_loss_LF);
% set(gca, 'YScale', 'log');
% xlabel('Standard deviation noise');
% ylabel('Test loss (-)');
% title('Robustness to noise')
% ylim(scale)
% 
% % Save the figure
% if save_plots
%     saveas(gcf, fullfile(figure_folder, 'box_plot.eps'), 'epsc');
%     saveas(gcf, fullfile(figure_folder, 'box_plot.png'));
% end