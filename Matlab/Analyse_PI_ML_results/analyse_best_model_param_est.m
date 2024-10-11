close all; clc; clear all;
plot_settings

save_plots = false;

%%
exp_name = 'PI_impact_trainingParam/Sim_13/';
file_name = 'LF_PITrue_Nr3_Nu1_Nseq50';

data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data_LF = readtable(data_name);
data_modelPar_LF = load(data_modelPar_name);
data_LF.L_test_tot = data_LF.L_test_NN + data_LF.L_test_Phys;

exp_name = 'PI_impact_trainingParam/Sim_13/';
file_name = 'DON_PITrue_Nr3_Nu1_Nseq50';

data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data_DON = readtable(data_name);
data_modelPar_DON = load(data_modelPar_name);
data_DON.L_test_tot = data_DON.L_test_NN + data_DON.L_test_Phys;

figure_folder = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end


%%
data_modelPar_DON = fun_calc_error(data_modelPar_DON);
data_modelPar_LF = fun_calc_error(data_modelPar_LF);

%filter the model with the best hyperparameters
c_DON = data_DON.eta_gain==0.005 & data_DON.init_gain==0.1;
% c_LF = data_LF.eta_gain==0.005 & data_LF.init_gain==0 & data_LF.phys_gain~=0;
c_LF = data_LF.eta_gain==0.005 & data_LF.init_gain==0.1;

errors_DON = [data_modelPar_DON.error_alpha(:,c_DON) data_modelPar_DON.error_beta(:,c_DON) ...
    data_modelPar_DON.error_gamma(:,c_DON) data_modelPar_DON.error_W(:,c_DON)];
errors_LF = [data_modelPar_LF.error_alpha(:,c_LF) data_modelPar_LF.error_beta(:,c_LF) ...
    data_modelPar_LF.error_gamma(:,c_LF) data_modelPar_LF.error_W(:,c_LF)];

x = length(data_modelPar_DON.error_alpha(:,c_DON));
name = [repmat("$\alpha$", 1,x)  repmat("$\beta$", 1,x) repmat("$\gamma$",  1,x) repmat("$W$",  1,x)];
errors = [errors_DON errors_LF];
names = [name name];
architect = [repmat("DON", 1, x*4) repmat("LF", 1, x*4)]; 

table_phys = table(architect,names,errors);

figure()
plt = groupedSpacedBoxchart(table_phys,'names','errors','architect');
hleg = legend(Location="best");
title(hleg,'Architect')
set(hleg, 'FontSize', 12);
xlabel('Physics model parameter');
ylabel('Normalised error (-)');
% title('Error estimation physics parameters')
grid on;
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');
% ylim([0 0.25])

if save_plots
    saveas(gcf, fullfile(figure_folder, 'PI_impact_error.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'PI_impact_error.png'));
end

% ylim([0 0.25])
% if save_plots
%     saveas(gcf, fullfile(figure_folder, 'zoom_PI_impact_error.eps'), 'epsc');
%     saveas(gcf, fullfile(figure_folder, 'zoom_PI_impact_error.png'));
% end

%% For table in report
% Calculate means
mean_DON = mean(errors_DON);
mean_LF = mean(errors_LF);

% Calculate standard deviations
std_DON = std(errors_DON);
std_LF = std(errors_LF);

% Display the results with labels
fprintf('Mean of errors for DON: %.4f\n', mean_DON);
fprintf('Mean of errors for LF: %.4f\n', mean_LF);
fprintf('Standard deviation of errors for DON: %.4f\n', std_DON);
fprintf('Standard deviation of errors for LF: %.4f\n', std_LF);