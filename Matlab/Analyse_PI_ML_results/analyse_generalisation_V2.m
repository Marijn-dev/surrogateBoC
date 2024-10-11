close all; clc; clear all;
plot_settings

save_plots = true;

global customColors

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,3,2,4,5,6], :);

%% Load results for LF and DON
exp_name_LF = 'Generalisation_LF/Sim_13/';
file_name_LF = 'LF_PITrue_Nr3_Nu1_Nseq100';
simN_LF = '13';
architect_LF = 'LF';

exp_name_DON = 'Generalisation_DON/Sim_13/';
file_name_DON = 'DON_PITrue_Nr3_Nu1_Nseq100';
simN_DON = '13';
architect_DON = 'DON';

% Load data for LF
data_name_LF = ['Results_sim/' exp_name_LF file_name_LF '_test_samples.csv'];
data_LF = readtable(data_name_LF);

% Load data for DON
data_name_DON = ['Results_sim/' exp_name_DON file_name_DON '_test_samples.csv'];
data_DON = readtable(data_name_DON);

figure_folder_LF = fullfile(['Results_sim/' exp_name_LF], 'Figures_V2');
figure_folder_DON = fullfile(['Results_sim/' exp_name_DON], 'Figures_V2');

if ~exist(figure_folder_LF, 'dir')
    mkdir(figure_folder_LF);
end
if ~exist(figure_folder_DON, 'dir')
    mkdir(figure_folder_DON);
end

%% Plot the losses for the different input distributions (LF and DON)
x_LF = length(data_LF.l_data);
x_DON = length(data_DON.l_data);

plot_y_LF = [data_LF.l_data; data_LF.l_phys; data_LF.l_data_sin; data_LF.l_phys_sin];
plot_y_DON = [data_DON.l_data; data_DON.l_phys; data_DON.l_data_sin; data_DON.l_phys_sin];

loss_type_LF = [repmat("Data", 1,x_LF)  repmat("Physics", 1,x_LF) repmat("Data", 1,x_LF)  repmat("Physics", 1,x_LF)]';
loss_type_DON = [repmat("Data", 1,x_DON)  repmat("Physics", 1,x_DON) repmat("Data", 1,x_DON)  repmat("Physics", 1,x_DON)]';

input_distr_LF = [repmat("$P_u$", 1,2*x_LF) repmat("$Q_u$", 1,2*x_LF)]';
input_distr_DON = [repmat("$P_u$", 1,2*x_DON) repmat("$Q_u$", 1,2*x_DON)]';

table_gener_LF = table(plot_y_LF,loss_type_LF,input_distr_LF);
table_gener_DON = table(plot_y_DON,loss_type_DON,input_distr_DON);

figure()
subplot(1,2,1);
plt_LF = groupedSpacedBoxchart(table_gener_LF,'input_distr_LF','plot_y_LF','loss_type_LF');
hleg_LF = legend('Location', 'southeast');
set(gca, 'YScale', 'log');
title(hleg_LF,'$\mathcal{L}$-term')
set(hleg_LF, 'FontSize', 12);
ylim([5*10^-4 20])
xlabel('Input distr. (LF)');
ylabel('Test loss terms (-)');
% title('LF Generalisation');
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');

subplot(1,2,2);
plt_DON = groupedSpacedBoxchart(table_gener_DON,'input_distr_DON','plot_y_DON','loss_type_DON');
hleg_DON = legend('Location', 'southeast');
set(gca, 'YScale', 'log');
title(hleg_DON,'$\mathcal{L}$-term')
set(hleg_DON, 'FontSize', 12);
ylim([5*10^-4 20])
xlabel('Input distr. (DON)');
ylabel('Test loss terms (-)');
% title('DON Generalisation');
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder_LF, [architect_LF '_vs_' architect_DON '_Gener_input_distr.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder_LF, [architect_LF '_vs_' architect_DON '_Gener_input_distr.png']));
end

%% Plot the trajectories for both LF and DON
i = '3';

% Plot pulse
traj_name_LF_pulse = ['Results_sim/' exp_name_LF simN_LF '_' architect_LF '_traj_pulse_' i '.mat'];
traj_LF_pulse = load(traj_name_LF_pulse);
traj_name_DON_pulse = ['Results_sim/' exp_name_DON simN_DON '_' architect_DON '_traj_pulse_' i '.mat'];
traj_DON_pulse = load(traj_name_DON_pulse);

% Plot LF vs DON pulse
plot_test_sample(traj_LF_pulse.t,traj_LF_pulse.u, traj_LF_pulse.r_est,traj_DON_pulse.r_est,traj_LF_pulse.r)

if save_plots
    saveas(gcf, fullfile(figure_folder_LF, [architect_LF '_vs_' architect_DON '_traj_pulse.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder_LF, [architect_LF '_vs_' architect_DON '_traj_pulse.png']));
end

% Plot sin
traj_name_LF_sin = ['Results_sim/' exp_name_LF simN_LF '_' architect_LF '_traj_sinusoidal_' i '.mat'];
traj_LF_sin = load(traj_name_LF_sin);
traj_name_DON_sin = ['Results_sim/' exp_name_DON simN_DON '_' architect_DON '_traj_sinusoidal_' i '.mat'];
traj_DON_sin = load(traj_name_DON_sin);

% Plot LF vs DON sinusoidal
plot_test_sample(traj_LF_sin.t,traj_LF_sin.u, traj_LF_sin.r_est,traj_DON_sin.r_est,traj_LF_sin.r)


if save_plots
    saveas(gcf, fullfile(figure_folder_LF, [architect_LF '_vs_' architect_DON '_traj_sin.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder_LF, [architect_LF '_vs_' architect_DON '_traj_sin.png']));
end


%% Helper function
function plot_test_sample(t_test, u, r_LF,r_DON, r)
global customColors

plot_settings

% Number of rows for subplots
N_r = size(r_LF, 2);

% Create the figure
fig = figure(); % 'Position', [1, 1, 0.4*screenSize(3), 0.6*screenSize(4)]);

% Define colors for u, r_LF, r_DON, and r
color_u = customColors(1, :);       % Color for u(t) (input in mA)
color_r_LF = customColors(3, :);    % Color for r_LF (LF prediction in Hz)
color_r_DON = customColors(4, :);   % Color for r_DON (DON prediction in Hz)
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
    plot(t_test, r_LF(:, i),'--', 'Color', color_r_LF, 'DisplayName', ['$\hat{r}_{LF_' num2str(i) '}(t)$'], 'LineWidth', 1.5);
    plot(t_test, r_DON(:, i),'--', 'Color', color_r_DON, 'DisplayName', ['$\hat{r}_{DON_' num2str(i) '}(t)$'], 'LineWidth', 1.5);
    ylabel(['$r_' num2str(i) '(t) \, \mathrm{(Hz)}$'], 'Interpreter', 'latex'); % Set ylabel color to black
    set(gca, 'YColor', 'black'); % Set color for left y-axis tick marks

    % Set right axis for u (input in mA)
    yyaxis right
    plot(t_test, u(:, i), 'Color', color_u, 'DisplayName', ['$u_' num2str(i) '(t)$'], 'LineWidth', 1.5);
    ylabel(['$u_' num2str(i) '(t) \, \mathrm{(mA)}$'], 'Interpreter', 'latex'); % Set ylabel color to black
    set(gca, 'YColor', 'black'); % Set color for left y-axis tick marks
    
    % Store y-axis limits from the first subplot
    if i == 1
        y_right_limits = ylim; % Store the right y-axis limits from the first subplot
    else
        ylim(y_right_limits); % Apply the stored limits to all subsequent subplots
    end
    
    % Grid and settings
    grid on;
end

xlabel('Time (s)');

% Create a combined legend
legend('$r(t)$', '$r_{LF}(t)$', '$r_{DON}(t)$','$u(t)$', 'Location', 'southeast', 'Interpreter', 'latex');

end
