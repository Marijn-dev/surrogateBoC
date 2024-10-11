close all; clc; clear all;
plot_settings

save_plots = false;

global customColors

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,3,2,4,5,6], :);

%% Load results for LF and DON
exp_name = 'u_window/Sim_7/';
file_name = '3_LF_PITrue_Nr3_Nu1_Nseq50';
architect = 'LF';
exp_num = '3';

% exp_name = 'Sim_1_traj_01sec/';
% file_name = '1_LF_PITrue_Nr3_Nu1_Nseq50';
% architect = 'LF';
% exp_num = '1';

data_name = ['Results_exp_data/' exp_name file_name '_training_logs.mat'];
data_train = load(data_name);

figure_folder = fullfile(['Results_exp_data/' exp_name], 'Figures_tests');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

%% Plot the losses
plot_training_validation_losses(data_train.loss_train_NN, data_train.loss_train_PHYS, data_train.loss_val_NN, data_train.loss_val_PHYS, data_train.gain)

if save_plots
    saveas(gcf, fullfile(figure_folder, [exp_num '_losses.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder, [exp_num '_losses.png']));
end
%% Plot the trajectories for both LF and DON
i = '2';

% Plot pulse
traj_name = ['Results_exp_data/' exp_name exp_num '_LF_traj_exp_' i '.mat'];
traj = load(traj_name);


% Plot LF vs DON pulse
plot_test_sample(traj.t,traj.u, traj.r_est,traj.r)

if save_plots
    saveas(gcf, fullfile(figure_folder, ['1_LF_traj_exp_' i '.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder, ['1_LF_traj_exp_' i '.png']));
end


%% Helper function
function plot_test_sample(t_test, u, r_est, r)
global customColors

plot_settings

% Number of rows for subplots
N_r = size(r_est, 2);

% Create the figure
fig = figure(); % 'Position', [1, 1, 0.4*screenSize(3), 0.6*screenSize(4)]);

% Define colors for u, r_LF, r_DON, and r
color_u = customColors(1, :);       % Color for u(t) (input in mA)
color_r_est = customColors(3, :);    % Color for r_LF (LF prediction in Hz)
color_r = customColors(2, :);       % Color for r (true states in Hz)

% Store the right y-axis limits (u-axis limits)
y_right_limits = [];

% Plot firing rates with dual y-axes
for i = 1:N_r
    subplot(N_r, 1, i);
    
    % Set left axis for r (frequencies in Hz)
    yyaxis left
    plot(t_test, r(:, i), '-', 'Color', color_r, 'DisplayName', ['$x_' num2str(i) '(t)$'], 'LineWidth', 1.5);
    hold on;
    plot(t_test, r_est(:, i),'--', 'Color', color_r_est, 'DisplayName', ['$\hat{x}_{LF_' num2str(i) '}(t)$'], 'LineWidth', 1.5);
    ylabel(['$r_' num2str(i) '(t) \, \mathrm{(Hz)}$'], 'Interpreter', 'latex'); % Set ylabel color to black
    set(gca, 'YColor', 'black'); % Set color for left y-axis tick marks

    % Set right axis for u (input in mA)
    yyaxis right
    plot(t_test, u(:, i), 'Color', color_u, 'DisplayName', ['$u_' num2str(i) '(t)$'], 'LineWidth', 1.5);
    ylabel(['$\bar{u}_' num2str(i) '(t) \, (\mu A)$'], 'Interpreter', 'latex'); % Set ylabel color to black
    set(gca, 'YColor', 'black'); % Set color for left y-axis tick marks
    ylim([-0.01 4])

    % Grid and settings
    grid on;
end

xlabel('Time (s)');

% Create a combined legend
% legend('$r(t)$', '$\hat{r}_{\mathrm{LF}}(t)$', '$u(t)$', 'Location', 'southeast', 'Interpreter', 'latex');
legend('$r(t)$', '$\hat{r}_{\mathrm{LF}}(t)$', '$\bar{u}(t)$', 'Location', 'southeast', 'Interpreter', 'latex');

end


%%% Plotting the data
function plot_training_validation_losses(log_train_data, log_train_phys, log_val_data, log_val_phys, log_gain)
    plot_settings    
    
    % Create a figure with specified position
    % fig = figure('Position', [1, 1, 0.5*screenSize(3), 0.6*screenSize(4)]);
    fig = figure();
    
    % Plot training and validation losses
    subplot(1, 2, 1);
    hold on;
    plot(1:length(log_train_data), log_train_data, 'Color', colours(1));
    plot(1:length(log_train_phys), log_train_phys, 'Color', colours(2));
    plot(1:length(log_val_data), log_val_data, 'Color', colours(3));
    plot(1:length(log_val_phys), log_val_phys, 'Color', colours(4));
    
    legendLabels = ["Data, train"; "Phys, train"; "Data, val"; "Phys, val"];
    hleg = legend(legendLabels,'Location', 'best');
    title(hleg,'$\mathcal{L}$-term, dataset')
    set(hleg, 'FontSize', 12);

    hold off;
    xlim([1, length(log_train_data)]);
    xlabel('Epoch (-)');
    ylabel('Loss value (-)');
    % title('Losses during training');
    % legend;
    grid on;
    set(gca, 'YScale', 'log');

    % Plot gain
    subplot(1, 2, 2);
    plot(1:length(log_gain), log_gain, 'DisplayName', 'Gain');
    xlabel('Epoch (-)');
    ylabel('Physics gain $\lambda$ (-)', 'Interpreter', 'latex');
    % title('Physics gain');
    xlim([1, length(log_train_data)]);
    grid on;
end
