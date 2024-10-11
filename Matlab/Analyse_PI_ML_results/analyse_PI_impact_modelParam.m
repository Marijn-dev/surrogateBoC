close all; clc; clear all;
plot_settings

save_plots = true;

%% Load results
exp_name = 'PI_impact_modelParam/Sim_13/';
file_name = 'LF_PITrue_Nr3_Nu1_Nseq50';

data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data = readtable(data_name);
data_modelPar = load(data_modelPar_name);
data.L_test_tot = data.L_test_NN + data.L_test_Phys;

figure_folder = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

%% Plotting
x = length(data.L_test_NN);

plot_y = [data.L_test_NN; data.L_test_Phys];
loss_type = [repmat("Data", 1,x)  repmat("Physics", 1,x)]';
paramVersion = [data.combi; data.combi]+1;
table_phys = table(plot_y,loss_type,paramVersion);

figure()
plt = groupedSpacedBoxchart(table_phys,'paramVersion','plot_y','loss_type');
hleg = legend('Location', 'southeast');
set(gca, 'YScale', 'log');
title(hleg,'$\mathcal{L}$-term')
set(hleg, 'FontSize', 12);
ylim([6*10^-4 0.3])
xlabel('BoC model version');
ylabel('Test loss value (-)');
% title('Loss result for different BoC models')
grid on;

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, 'loss_phys_par.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'loss_phys_par.png'));
end

%% Prepare the model parameter error table
data_modelPar = fun_calc_error(data_modelPar);
errors = [data_modelPar.error_alpha data_modelPar.error_beta ...
    data_modelPar.error_gamma data_modelPar.error_W];

x = length(data_modelPar.error_alpha);
name = [repmat("$\alpha$", 1,x)  repmat("$\beta$", 1,x) repmat("$\gamma$",  1,x) repmat("$W$",  1,x)];
combi = [data.combi; data.combi; data.combi; data.combi]'+1;
table_phys = table(name,errors,combi);

figure()
plt = groupedSpacedBoxchart(table_phys,'name','errors','combi');
hleg = legend(Location="northeast");
title(hleg,'BoC model')
set(hleg, 'FontSize', 12);
xlabel('BoC model parameter');
ylabel('Normalised noise (-)');
% title('Error estimation for different BoC models')
grid on;
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, 'error_phys_par.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'error_phys_par.png'));
end