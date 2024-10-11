close all; clc; clear all;
plot_settings

save_plots = true;
%%
loss_type_plot = 'L_test_NN';
% loss_type_plot = 'L_test_tot';
% loss_type_plot = 'L_test_Phys';

range = [3*10^-3 2]; 

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1, 2, 3, 6, 4, 5], :);

%% Plot the DON results
exp_name = 'hyperParamSearch_DON/Sim_13/';
file_name = 'DON_PITrue_Nr3_Nu1_Nseq50';
cat_group = 'n_hidden';

data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data_DON = readtable(data_name);
data_modelPar_DON = load(data_modelPar_name);
data_DON.L_test_tot = data_DON.L_test_NN + data_DON.L_test_Phys;

figure_folder_DON = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder_DON, 'dir')
    mkdir(figure_folder_DON);
end

% Plot a single loss term
% figure()
% plt = groupedSpacedBoxchart(data_DON,'eta',loss_type_plot,cat_group);
% hleg = legend(categorical(unique(data_DON.n_hidden)), 'Location', 'best');
% title(hleg,'\# hidden states')
% set(hleg, 'FontSize', 12);
% set(gca, 'YScale', 'log');
% ylim(range)
% xlabel('Learning rate');
% ylabel('Data Test Loss');
% % ylabel('Physics Test Loss');
% title('Hyperparameter search PI-DeepONet')
% grid on;

% Plot the combined plots
figure()
plt = groupedSpacedBoxchart(data_DON,'eta','L_test_NN',cat_group);
hold on 
plt = groupedSpacedBoxchart(data_DON,'eta','L_test_Phys',cat_group);
legendLabels = [string(unique(data_DON.n_hidden)) + ", Data"; string(unique(data_DON.n_hidden))+ ", Phys"];
hleg = legend(legendLabels,'Location', 'northeast', 'NumColumns', 2);
title(hleg,'\# h-states, $\mathcal{L}$-term')
set(gca, 'YScale', 'log');
set(hleg, 'FontSize', 12);
xlabel('Learning rate $\eta$');
ylabel('Test loss value (-)');
% title('Hyperparameter search PI-DeepONet')
set(gca, 'ColorOrder', customColors, 'NextPlot', 'replacechildren');
ylim(range)

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder_DON, 'DON_box_both_terms.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder_DON, 'DON_box_both_terms.png'));
end

%% Plot the LF results
exp_name = 'hyperParamSearch_LF/Sim_13/';
file_name = 'LF_PITrue_Nr3_Nu1_Nseq50';
cat_group = 'control_rnn_size';

data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data_LF = readtable(data_name);
data_modelPar_LF = load(data_modelPar_name);
data_LF.L_test_tot = data_LF.L_test_NN + data_LF.L_test_Phys;

figure_folder_LF = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder_LF, 'dir')
    mkdir(figure_folder_LF);
end

% Plot a single loss term
% figure()
% plt = groupedSpacedBoxchart(data_LF,'eta',loss_type_plot,cat_group);
% hleg = legend(categorical(unique(data_LF.control_rnn_size)), 'Location', 'best');
% title(hleg,'\# hidden states')
% set(hleg, 'FontSize', 12);
% set(gca, 'YScale', 'log');
% ylim(range)
% xlabel('Learning rate');
% ylabel('Data Test Loss');
% % ylabel('Physics Test Loss');
% title('Hyperparameter search PI-Learning Flow')
% grid on;


% plot the combined plots
figure()
plt = groupedSpacedBoxchart(data_LF,'eta','L_test_NN',cat_group);
hold on 
plt = groupedSpacedBoxchart(data_LF,'eta','L_test_Phys',cat_group);
legendLabels = [string(unique(data_LF.control_rnn_size)) + ", Data"; string(unique(data_LF.control_rnn_size))+ ", Phys"];
hleg = legend(legendLabels,'Location', 'northeast', 'NumColumns', 2);
title(hleg,'\# h-states, $\mathcal{L}$-term')
set(gca, 'YScale', 'log');
set(hleg, 'FontSize', 12);
xlabel('Learning rate $\eta$');
ylabel('Test loss value (-)');
% title('Hyperparameter search PI-Learning Flow')
set(gca, 'ColorOrder', customColors, 'NextPlot', 'replacechildren');
ylim(range)

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder_LF, 'LF_box_both_terms.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder_LF, 'LF_box_both_terms.png'));
end

%%
data_modelPar_DON = fun_calc_error(data_modelPar_DON);
data_modelPar_LF = fun_calc_error(data_modelPar_LF);

%filter the model with the best hyperparameters
c_DON = data_DON.n_hidden==60 & data_DON.eta==0.01;
c_LF = data_LF.control_rnn_size==24 & data_LF.eta==0.01;

errors_DON = [data_modelPar_DON.error_alpha(:,c_DON) data_modelPar_DON.error_beta(:,c_DON) ...
    data_modelPar_DON.error_gamma(:,c_DON) data_modelPar_DON.error_W(:,c_DON)];
errors_LF = [data_modelPar_LF.error_alpha(:,c_LF) data_modelPar_LF.error_beta(:,c_LF) ...
    data_modelPar_LF.error_gamma(:,c_LF) data_modelPar_LF.error_W(:,c_LF)];

x = length(data_modelPar_DON.error_alpha(:,c_LF));
name = [repmat("$\alpha$", 1,x)  repmat("$\beta$", 1,x) repmat("$\gamma$",  1,x) repmat("$W$",  1,x)];
errors = [errors_DON errors_LF];
names = [name name];
architect = [repmat("DON", 1, x*4) repmat("LF", 1, x*4)]; 

table_phys = table(architect,names,errors);


figure()
plt = groupedSpacedBoxchart(table_phys,'names','errors','architect');
hleg = legend(Location="northeast");
title(hleg,'Architect')
set(hleg, 'FontSize', 12);
xlabel('Physics model parameter');
ylabel('Normalised error (-)');
% title('Error estimation physics parameters')
grid on;
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');
% ylim([0 0.4])

if save_plots
    saveas(gcf, fullfile(figure_folder_DON, 'Phys_par_error.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder_DON, 'Phys_par_error.png'));
    saveas(gcf, fullfile(figure_folder_LF, 'Phys_par_error.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder_LF, 'Phys_par_error.png'));
end
