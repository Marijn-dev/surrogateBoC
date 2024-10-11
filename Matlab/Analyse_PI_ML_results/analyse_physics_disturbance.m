close all; clc; clear all;
plot_settings

save_plots = false;

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,2,6,7,3,4,5], :);
close all

%% Load data
exp_name = 'physics_disturbance/Sim_13/';
file_name_DON = 'DON_PITrue_Nr3_Nu1_Nseq50';
file_name_LF = 'LF_PITrue_Nr3_Nu1_Nseq50';

data_name = ['Results_sim/' exp_name file_name_DON '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name_DON '_phys_par.mat'];
data_DON = readtable(data_name);
data_modelPar_DON = load(data_modelPar_name);
data_DON.L_test_tot = data_DON.L_test_NN + data_DON.phys_gain.*data_DON.L_test_Phys;

data_name = ['Results_sim/' exp_name file_name_LF '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name_LF '_phys_par.mat'];
data_LF = readtable(data_name);
data_modelPar_LF = load(data_modelPar_name);
data_LF.L_test_tot = data_LF.L_test_NN + data_LF.phys_gain.*data_LF.L_test_Phys;

figure_folder_DON = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder_DON, 'dir')
    mkdir(figure_folder_DON);
end

range = [3*10^-3 0.3]; 

%% Combine data in new table
x = length(data_DON.L_test_NN);
phys_dist = [data_DON.phys_dist; data_LF.phys_dist];
data_Loss = [data_DON.L_test_NN; data_LF.L_test_NN];
phys_Loss = [data_DON.L_test_Phys; data_LF.L_test_Phys];
tot_Loss = [data_DON.L_test_tot; data_LF.L_test_tot];
architect = [repmat("DON", x,1); repmat("LF", x,1)]; 

table_combined_result = table(architect,phys_dist,data_Loss,phys_Loss,tot_Loss);

%% Plot results

% Plot the combined plots
figure()
plt = groupedSpacedBoxchart(table_combined_result,'phys_dist','data_Loss','architect'); %tot_Loss
hold on 
plt = groupedSpacedBoxchart(table_combined_result,'phys_dist','phys_Loss','architect');

legendLabels = ["Data, " + string(unique(table_combined_result.architect)); "Phys, " + string(unique(table_combined_result.architect))];
hleg = legend(legendLabels,'Location', 'bestoutside');
title(hleg,'$\mathcal{L}$-term, Architect')
set(hleg, 'FontSize', 12);

% legendLabels = [string(unique(table_combined_result.architect)) + ", Data"; string(unique(table_combined_result.architect))+ ", Phys"];
% hleg = legend(legendLabels,'Location', 'best', 'NumColumns', 2);
% title(hleg,'Architect, $\mathcal{L}$-term')
% set(hleg, 'FontSize', 10);

ylim(range)
set(gca, 'YScale', 'log');
xlabel('Norm of the disturbance matrix $\left\Vert \delta W \right\Vert$ (-)');
ylabel('Test Loss (-)');
% title('Hyperparameter search PI-DeepONet')
set(gca, 'ColorOrder', customColors, 'NextPlot', 'replacechildren');

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder_DON, 'phys_dist_combi_loss.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder_DON, 'phys_dist_combi_loss.png'));
end


%% error of the parameters

data_modelPar_DON = fun_calc_error(data_modelPar_DON);
errors_DON = [data_modelPar_DON.error_alpha data_modelPar_DON.error_beta ...
    data_modelPar_DON.error_gamma data_modelPar_DON.error_W]';

data_modelPar_LF = fun_calc_error(data_modelPar_LF);
errors_LF = [data_modelPar_LF.error_alpha data_modelPar_LF.error_beta ...
    data_modelPar_LF.error_gamma data_modelPar_LF.error_W]';

x = length(data_modelPar_DON.error_alpha);
architect = [repmat("DON", 4*x,1); repmat("LF", 4*x,1)]; 
name = [repmat("$\alpha$", 1,x)  repmat("$\beta$", 1,x) repmat("$\gamma$",  1,x) repmat("$W$",  1,x)]';
phys_dist = [data_DON.phys_dist; data_DON.phys_dist; data_DON.phys_dist; data_DON.phys_dist];
phys_dist_tb = [phys_dist;phys_dist];
name_tb = [name; name];
error_tb = [errors_DON; errors_LF];
table_phys = table(name_tb,error_tb, architect,phys_dist_tb);


%% plot error results
scale = [0 0.43];

table_phys_c = table_phys(table_phys.architect=="DON",:);

figure()
plt = groupedSpacedBoxchart(table_phys_c,'name_tb','error_tb','phys_dist_tb');
hleg = legend('Location', 'northeast');
title(hleg,'$\left\Vert \delta W \right\Vert$')
set(hleg, 'FontSize', 12);
xlabel('BoC model parameter');
ylabel('Normalised error (-)');
% title('Error estimation physics parameters')
grid on;
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');
ylim(scale)

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder_DON, 'phys_dist_error_DON.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder_DON, 'phys_dist_error_DON.png'));
end

table_phys_c = table_phys(table_phys.architect=="LF",:);

figure()
plt = groupedSpacedBoxchart(table_phys_c,'name_tb','error_tb','phys_dist_tb');
hleg = legend('Location', 'northeast');
title(hleg,'$\left\Vert \delta W \right\Vert$')
set(hleg, 'FontSize', 12);
xlabel('BoC model parameter');
ylabel('Normalised error (-)');
% title('Error estimation physics parameters')
grid on;
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');
ylim(scale)

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder_DON, 'phys_dist_error_LF.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder_DON, 'phys_dist_error_LF.png'));
end