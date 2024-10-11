close all; clc; clear all;
plot_settings

save_plots = true;

%% Load results
exp_name = 'Exp_design/Sim_13/';
file_name = 'LF_PITrue_Nr3_Nu1_Nseq100';

data_name = ['Results_sim/' exp_name file_name '_results.csv'];
data_modelPar_name = ['Results_sim/' exp_name file_name '_phys_par.mat'];
data = readtable(data_name);
data_modelPar = load(data_modelPar_name);
data.L_test_tot = data.L_test_NN + data.phys_gain .* data.L_test_Phys;

figure_folder = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

%% Prepare the model parameter error table
data_modelPar = fun_calc_error(data_modelPar);
errors = [data_modelPar.error_alpha data_modelPar.error_beta ...
    data_modelPar.error_gamma data_modelPar.error_W]';

x = length(data_modelPar.error_alpha);
name = [repmat("$\alpha$", 1,x)  repmat("$\beta$", 1,x) repmat("$\gamma$",  1,x) repmat("$W$",  1,x)]';
tend = [data.tend; data.tend; data.tend; data.tend];
N_data_seq = [data.N_data_seq; data.N_data_seq; data.N_data_seq; data.N_data_seq];
table_phys = table(name,errors,tend,N_data_seq);

condition = table_phys.tend==20;
table_phys_filter = table_phys(condition,:); 

figure()
plt = groupedSpacedBoxchart(table_phys_filter,'name','errors','N_data_seq');
hleg = legend('Location', 'northeast');
title(hleg,'\# train sequences')
set(hleg, 'FontSize', 12);
xlabel('BoC model parameter');
ylabel('Normalised error (-)');
% title('Error estimation physics parameters')
grid on;
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, 'ExpDes_phys_par.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'ExpDes_phys_par.png'));
end

%% Make the heatmap
loss_type_plot = 'L_test_tot';
data_avg = groupsummary(data, {'N_data_seq', 'tend'}, 'median', loss_type_plot);

% Identify unique categories and subcategories
categories = unique(data_avg.N_data_seq);
subcategories = unique(data_avg.tend);

% Initialize the values matrix
values = zeros(length(categories), length(subcategories));

% Fill the values matrix with the average values
for i = 1:height(data_avg)
    category_idx = find(data_avg.N_data_seq(i)==categories);
    subcategory_idx = find(data_avg.tend(i)==subcategories);
    % values(category_idx, subcategory_idx) = data_avg.median_L_test_NN(i);
    values(category_idx, subcategory_idx) = data_avg.median_L_test_tot(i);
end

figure()
h = heatmap(subcategories, categories, values);
colormap sky %summer
set(gca,'ColorScaling','log') % Log scale
% clim(log([2*10^-2 2]));
xlabel('Horizon of a single sequence (s)')
ylabel('Number of training sequences (-)')
% annotation('textarrow',[1,1],[0.5,0.5],'string','Test loss (-)', 'Interpreter', 'latex',...
%       'HeadStyle','none','LineStyle','none','HorizontalAlignment','center','TextRotation',90);
annotation('textarrow',[0.99,0.99],[0.6,0.6],'string','Test loss (-)','HeadStyle','none','LineStyle','none','HorizontalAlignment','right','TextRotation',90,'Interpreter', 'latex');

% title('Test loss for different training set sizes')
h.NodeChildren(3).XAxis.Label.Interpreter = 'latex';
h.NodeChildren(3).YAxis.Label.Interpreter = 'latex';
h.NodeChildren(3).Title.Interpreter = 'latex';
h.FontSize = 12;
h.NodeChildren(3).Title.FontSize = 16;
h.NodeChildren(3).XAxis.FontSize = 16; 
h.NodeChildren(3).YAxis.FontSize = 16;


% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, 'ExpDes_heatmap.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'ExpDes_heatmap.png'));
end


%% Plotting
% scale = [3e-3 6e-1];
% loss_type_plot = 'L_test_Phys';
% 
% figure()
% plt = groupedSpacedBoxchart(data,'N_data_seq',loss_type_plot,'tend');
% hleg = legend(categorical(unique(data.tend)), 'Location', 'best');
% title(hleg,'$t_{end}$ (s)')
% set(hleg, 'FontSize', 12);
% set(gca, 'YScale', 'log');
% xlabel('Number of training sequences');
% ylabel('Physics loss term');
% title('Impact of training set size')
% ylim(scale)
% 
% % Save the figure
% if save_plots
%     saveas(gcf, fullfile(figure_folder, 'loss_phys.eps'), 'epsc');
%     saveas(gcf, fullfile(figure_folder, 'loss_phys.png'));
% end
% 
% loss_type_plot = 'L_test_NN';
% 
% figure()
% plt = groupedSpacedBoxchart(data,'N_data_seq',loss_type_plot,'tend');
% hleg = legend(categorical(unique(data.tend)), 'Location', 'best');
% set(gca, 'YScale', 'log');
% title(hleg,'$t_{end}$ (s)')
% set(hleg, 'FontSize', 12);
% xlabel('Number of training sequences');
% ylabel('Test Loss Data term');
% title('Impact of training set size')
% ylim(scale)
% 
% % Save the figure
% if save_plots
%     saveas(gcf, fullfile(figure_folder, 'loss_data.eps'), 'epsc');
%     saveas(gcf, fullfile(figure_folder, 'loss_data.png'));
% end
% 
% loss_type_plot = 'L_test_tot';
% 
% figure()
% plt = groupedSpacedBoxchart(data,'N_data_seq',loss_type_plot,'tend');
% hleg = legend(categorical(unique(data.tend)), 'Location', 'best');
% title(hleg,'$t_{end}$ (s)')
% set(hleg, 'FontSize', 12);
% set(gca, 'YScale', 'log');
% xlabel('Number of training sequences');
% ylabel('Test Loss Data + Physics terms');
% title('Impact of training set size')
% ylim(scale)
% 
% % Save the figure
% if save_plots
%     saveas(gcf, fullfile(figure_folder, 'loss_tot.eps'), 'epsc');
%     saveas(gcf, fullfile(figure_folder, 'loss_tot.png'));
% end
