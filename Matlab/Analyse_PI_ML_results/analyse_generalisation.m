close all; clc; clear all;
plot_settings

save_plots = false;

global customColors

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,3,2,4,5,6], :);

%% Load results
exp_name = 'Generalisation_LF/Sim_13/';
file_name = 'LF_PITrue_Nr3_Nu1_Nseq100';
simN = '13';
architect = 'LF';

% exp_name = 'Generalisation_DON/Sim_13/';
% file_name = 'DON_PITrue_Nr3_Nu1_Nseq100';
% simN = '13';
% architect = 'DON';

data_name = ['Results_sim/' exp_name file_name '_test_samples.csv'];
data = readtable(data_name);

figure_folder = fullfile(['Results_sim/' exp_name], 'Figures');
if ~exist(figure_folder, 'dir')
    mkdir(figure_folder);
end

%% Plot the losses for the different input distr
x = length(data.l_data);

plot_y = [data.l_data; data.l_phys; data.l_data_sin; data.l_phys_sin];
loss_type = [repmat("Data", 1,x)  repmat("Physics", 1,x) repmat("Data", 1,x)  repmat("Physics", 1,x)]';
input_distr = [repmat("$P_u$", 1,2*x) repmat("$Q_u$", 1,2*x)]';
table_gener = table(plot_y,loss_type,input_distr);

figure()
plt = groupedSpacedBoxchart(table_gener,'input_distr','plot_y','loss_type');
hleg = legend('Location', 'southeast');
set(gca, 'YScale', 'log');
title(hleg,'$\mathcal{L}$-term')
set(hleg, 'FontSize', 8);
ylim([5*10^-4 20])
xlabel('Input Distribution');
ylabel('Test loss value (-)');
% title('Generalisation for the input')
ax = gca;
set(ax.XAxis, 'TickLabelInterpreter', 'latex');

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, [architect 'Gener_input_distr.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder, [architect 'Gener_input_distr.png']));
end

%% Plot some of the trajectories
i = '4';
traj_name_pulse = ['Results_sim/' exp_name simN '_' architect '_traj_pulse_' i '.mat'];

i = '4';
traj_name_sin = ['Results_sim/' exp_name simN '_' architect '_traj_sinusoidal_' i '.mat'];

traj_pulse = load(traj_name_pulse);
traj_sin = load(traj_name_sin);

plot_test_sample(traj_pulse.t,traj_pulse.u, traj_pulse.r, traj_pulse.r_est)

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder, 'traj_pulse.eps'), 'epsc');
    saveas(gcf, fullfile(figure_folder, 'traj_pulse.png'));
end

plot_test_sample(traj_sin.t,traj_sin.u, traj_sin.r, traj_sin.r_est)

% Save the figure
if save_plots
    saveas(gcf, fullfile(figure_folder,[architect 'traj_sin.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder,[architect 'traj_sin.png']));
end

%% Plot the loss values

data_name = ['Results_sim/' exp_name '13_LF_PITrue_Nr3_Nu1_Nseq100_training_logs.mat'];
data_train = load(data_name);

n_epoch = length(data_train.loss_train_NN);

figure()
subplot(1,2,1)
plot(1:n_epoch,data_train.loss_train_NN,DisplayName='Train, Data')
hold on
plot(1:n_epoch,data_train.loss_train_PHYS,DisplayName='Train, Phys')
set(gca, 'YScale', 'log');
plot(1:n_epoch,data_train.loss_val_NN,DisplayName='Val, Data')
plot(1:n_epoch,data_train.loss_val_PHYS,DisplayName='Train, Phys')
xlabel('Epochs')
ylabel('Loss value (-)')
xlim([0 n_epoch])
hleg = legend('Location', 'northeast');
title(hleg,'$\mathcal{L}$-term')
set(hleg, 'FontSize', 10);

subplot(1,2,2)
plot(1:n_epoch,data_train.gain)
ylabel('Physics gain $\lambda$')
xlabel('Epochs')
xlim([0 n_epoch])

if save_plots
    saveas(gcf, fullfile(figure_folder,[architect 'loss_train.eps']), 'epsc');
    saveas(gcf, fullfile(figure_folder,[architect 'loss_train.png']));
end


%% helper functions
function plot_test_sample(t_test, u, r, r_pred)
    global customColors

    plot_settings

    % defaultColors = get(gca, 'ColorOrder');
    % customColors = defaultColors([1,3,2,4,5,6], :);
    
    % Number of rows for subplots
    N_r = size(r, 2);

    % Create the figure
    fig = figure('Position', [1, 1, 0.4*screenSize(3), 0.6*screenSize(4)]);
    
    % Plot firing rates
    for i = 1:N_r
        subplot(N_r, 1, i);
        plot(t_test, u(:, i));
        hold on;
        plot(t_test, r(:, i));
        plot(t_test, r_pred(:, i), '--');
        % title("Neural Mass " + num2str(i))
        % ylabel('r [Hz]');
        ylabel(['$r_' num2str(i) ' \, \mathrm{[Hz]}$'], 'Interpreter', 'latex');
        grid on;
        set(gca, 'ColorOrder', customColors, 'NextPlot', 'replacechildren');

    end
    xlabel('time [s]');
    legend('Input','True states','Prediction', 'Location','southeast');

end
