close all; clc; clear all;
plot_settings

save_plots = false;

defaultColors = get(gca, 'ColorOrder');
% customColors = defaultColors([1, 2, 3, 6, 4, 5], :);
customColors = defaultColors([1,2,6,7,3,4,5], :);

close all
%% Plot the DON results
exp_name_DON = 'Upscaling_DON/Sim_13/';
exp_name_LF = 'Upscaling_LF/Sim_13/';
file_name_DON = 'DON_PITrue_Nr30_Nu10_Nseq50';
file_name_LF = 'LF_PITrue_Nr30_Nu10_Nseq50';

data_name = ['Results_sim/' exp_name_DON file_name_DON '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name_DON file_name_DON '_phys_par.mat'];
data_DON = readtable(data_name);
data_modelPar_DON = load(data_modelPar_name);
data_DON.L_test_tot = data_DON.L_test_NN + data_DON.phys_gain.*data_DON.L_test_Phys;

data_name = ['Results_sim/' exp_name_LF file_name_LF '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name_LF file_name_LF '_phys_par.mat'];
data_LF = readtable(data_name);
data_modelPar_LF = load(data_modelPar_name);
data_LF.L_test_tot = data_LF.L_test_NN + data_LF.phys_gain.*data_LF.L_test_Phys;

figure_folder = fullfile(['Results_sim/' exp_name_LF], 'Figures');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

%%
x = length(data_DON.L_test_NN);
Nr = [data_DON.N_r; data_LF.N_r];
Nu = [data_DON.N_u; data_LF.N_u];
data_Loss = [data_DON.L_test_NN; data_LF.L_test_NN];
phys_Loss = [data_DON.L_test_Phys; data_LF.L_test_Phys];
tot_Loss = [data_DON.L_test_tot; data_LF.L_test_tot];
architect = [repmat("DON", x,1); repmat("LF", x,1)]; 

table_combined_result = table(architect,Nr,Nu,data_Loss,phys_Loss,tot_Loss);

%%
% Plot the combined plots
figure()
plt = groupedSpacedBoxchart(table_combined_result,'Nr','data_Loss','architect');
hold on 
plt = groupedSpacedBoxchart(table_combined_result,'Nr','phys_Loss','architect');
% legendLabels = [string(unique(table_combined_result.architect)) + ", Data"; string(unique(table_combined_result.architect))+ ", Phys"];
legendLabels = ["Data, " + string(unique(table_combined_result.architect)); "Phys, " + string(unique(table_combined_result.architect))];
hleg = legend(legendLabels,'Location','northwest');
% title(hleg,'Architect, $\mathcal{L}$-term')
title(hleg,'$\mathcal{L}$-term, Architect')
set(hleg, 'FontSize', 12);
ylim([0 10])
% title('Upscaling results')
set(gca, 'YScale', 'log');
xlabel('Number of firing rate units M (-)');
ylabel('Test loss value(-)');
set(gca, 'ColorOrder', customColors, 'NextPlot', 'replacechildren');

if save_plots
    saveas(gcf, fullfile(figure_folder, 'acc_upscale.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'acc_upscale.png'));
end

%% visualise computation time
[meansDON,stdsDON,mediansDON,optionsDON] = fun_calc_mean(data_DON.N_r,data_DON.train_time/60);
[meansLF,stdsLF,mediansLF,optionsLF] = fun_calc_mean(data_LF.N_r,data_LF.train_time/60);

figure();
plot(optionsDON, meansDON,'-*','Color', customColors(1,:));
hold on
plot(optionsDON, meansLF,'-*','Color', customColors(2,:));
fill([optionsDON', fliplr(optionsDON')], [meansDON'+stdsDON'; flipud(meansDON'-stdsDON')], customColors(1,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill([optionsDON', fliplr(optionsDON')], [meansLF'+stdsLF'; flipud(meansLF'-stdsLF')], customColors(2,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
hleg = legend(categorical(unique(table_combined_result.architect)), 'Location', 'northwest');
set(hleg, 'FontSize', 12);
title(hleg,'Architect')
xlabel('Number of firing rate units M (-)');
ylabel('Learning time (min.)');
% title('Upscaling results')
xlim([0 30])
% set(gca, 'YScale', 'log');
% ylim(scale)

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, 'compt_time_upscale.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'compt_time_upscale.png'));
end
