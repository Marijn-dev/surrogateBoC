close all, clc, clear all
plot_settings
global customColors

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,3,2,4,5,6], :);
%% 
% testlog = load('D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\Matlab_FR_models\V2_trained_models/6_LF_PITrue_Nr3_Nu1_Nseq40_test_results'); 
% trainlog = load('D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\Matlab_FR_models\V2_trained_models/6_LF_PITrue_Nr3_Nu1_Nseq40_training_logs');

folder = 'D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\Matlab_analysis_PIML\Results_sim\Simple_sim\';
test_LF = load([folder '13_LF_PITrue_Nr3_Nu1_Nseq50_test_results']);
train_LF = load([folder '13_LF_PITrue_Nr3_Nu1_Nseq50_training_logs']);
test_DON = load([folder '13_DON_PITrue_Nr3_Nu1_Nseq50_test_results']);
train_DON = load([folder '13_DON_PITrue_Nr3_Nu1_Nseq50_training_logs']);

N_r = 3;

W = [0 0 0.8 0 1.5 0];
alpha = [1.8 0.8 2.5];
beta = [25 30 20];
gamma = [7 5 5];

true_params{1} = alpha;
true_params{2} = beta;
true_params{3} = gamma;
true_params{4} = W;

%% plot training losses
[LtrainNN,LtrainPHys,LtrainTot,LvalNN,LvalPhys,LvalTot,gain,phys_par] = load_trainLog(train_LF);
plot_training_validation_losses(LtrainNN, LtrainPHys, LvalNN, LvalPhys, gain)

[LtrainNN,LtrainPHys,LtrainTot,LvalNN,LvalPhys,LvalTot,gain,phys_par] = load_trainLog(train_DON);
plot_training_validation_losses(LtrainNN, LtrainPHys, LvalNN, LvalPhys, gain)

%% phys errors
phys_par_error = calc_par_error(phys_par,true_params);

%% Plot trajectoreis
[t,u,r,r_LF,~,~] = load_testLog(test_LF);
[~, ~, ~,r_DON,~,~] = load_testLog(test_DON);

plot_test_sample_LF_DON(t, u, r_LF,r_DON, r)


%% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Loading the data
function [t,u,r,r_e,loss_data,loss_phys] = load_testLog(nameTest)
    t = nameTest.t_batch;
    u = nameTest.u_batch;
%     r0 = nameTest.r0_batch;
    r = nameTest.r_batch;
    r_e = nameTest.r_est;
%     r_p = nameTest.r_phys;
%     drdt = nameTest.dr_dt;
    loss_data = nameTest.loss_data;
    loss_phys = nameTest.loss_phys;
end 

function [LtrainNN,LtrainPHys,LtrainTot,LvalNN,LvalPhys,LvalTot,gain,phys_par] = load_trainLog(nameTrain)
    LtrainNN = nameTrain.loss_train_NN;
    LtrainPHys = nameTrain.loss_train_PHYS;
    LtrainTot = nameTrain.loss_train_tot;
    LvalNN = nameTrain.loss_val_NN;
    LvalPhys = nameTrain.loss_val_PHYS;
    LvalTot = nameTrain.loss_val_tot;
    gain = nameTrain.gain;
    phys_par = nameTrain.Phys_par;
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
    hleg = legend(legendLabels,'Location', 'northwest');
    title(hleg,'$\mathcal{L}$-term, dataset')
    set(hleg, 'FontSize', 12);

    hold off;
    xlim([1, length(log_train_data)]);
    xlabel('Epoch (-)');
    ylabel('Loss value (-)');
    % title('Losses during training');
    legend;
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


%%% Compute the physics parameter errors
function phys_par_error = calc_par_error(phys_par,true_params)   
    % Get the dimensions of the phys_par cell array
    [numRows, numCols] = size(phys_par);

    % Initialize the error cell array with the same size as phys_par
    phys_par_error = cell(numCols);

    % Iterate over each cell in phys_par
    for col = 1:numCols
        error_matrix = zeros(numRows,length(true_params{col}));
        
        for row = 1:numRows
            % Get the current estimation matrix
            est_params = phys_par{row, col};
            
            % Get the corresponding true parameters for this column
            true_params_col = true_params{col};
            
            % Calculate the error
%             error_matrix(row,:) = est_params - true_params_col;
            error_matrix(row,:) = est_params ;

        end
        
        phys_par_error{col} = error_matrix;
    end
end

function plot_test_sample(t_test, u, r, r_pred)
    plot_settings
    
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
        title(["Neural Mass " + num2str(i)])
        ylabel('r [Hz]');
        grid on;
    end
    xlabel('time [s]');
    legend('Input','True states','Prediction');
end

function plot_phys_par_estimates(phys_par_error)
    plot_settings
    % Get the number of rows and columns in phys_par
    numCols = length(phys_par_error);
    param_names = {"Alpha", 'Beta', 'Gamma', 'Synap. weights'};
    % Create a new figure
    fig = figure('Position', [1, 1, 0.7*screenSize(3), 0.7*screenSize(4)]);  
    for col = 1:numCols
        subplot(2, 2, col); % Create a 2x2 grid of subplots
        hold on;
        plot(phys_par_error{col})
        hold off;
        xlabel('Epoch [-]');
        ylabel('Estimation error');
        title(["Parameter: " + param_names(col)]);
        grid on;
    end
end

function plot_test_sample_LF_DON(t_test, u, r_LF,r_DON, r)
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
    h1 = plot(t_test, r(:, i), '-', 'Color', color_r, 'DisplayName', ['$x_' num2str(i) '(t)$'], 'LineWidth', 1.5);
    hold on;
    h2 = plot(t_test, r_LF(:, i),'--', 'Color', color_r_LF, 'DisplayName', ['$\varphi_{' num2str(i) '}(t,r_0,u)$'], 'LineWidth', 1.5);
    h3 = plot(t_test, r_DON(:, i),'--', 'Color', color_r_DON, 'DisplayName', ['$G_{' num2str(i) '}G(r_x^k,u)(t)$'], 'LineWidth', 1.5);
    ylabel(['$r_' num2str(i) '(t) \, \mathrm{(Hz)}$'], 'Interpreter', 'latex'); % Set ylabel color to black
    set(gca, 'YColor', 'black'); % Set color for left y-axis tick marks

    % Set right axis for u (input in mA)
    yyaxis right
    h4 = plot(t_test, u(:, i), 'Color', color_u, 'DisplayName', ['$u_' num2str(i) '(t)$'], 'LineWidth', 1.5);
    ylabel(['$u_' num2str(i) '(t) \, \mathrm{(\mu A)}$'], 'Interpreter', 'latex'); % Set ylabel color to black
    set(gca, 'YColor', 'black'); % Set color for left y-axis tick marks
    % set(gca, 'YColor', color_u); % Set color for left y-axis tick marks

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
% legend('$r(t)$', '$\varphi(t,r_0,u)$', '$G(r,u)(t)$','$u(t)$', 'Location', 'southeast', 'Interpreter', 'latex','NumColumns',2);
% legend('$r(t)$', '$\hat{r}_{\mathrm{LF}}(t)$', '$\hat{r}_{\mathrm{DON}}(t)$','$u(t)$', 'Location', 'southeast', 'Interpreter', 'latex','NumColumns',2);
legend([h1 h4 h2 h3 ],'$r(t)$','$u(t)$','$\hat{r}_{\mathrm{LF}}(t)$', '$\hat{r}_{\mathrm{DON}}(t)$', 'Location', 'southeast', 'Interpreter', 'latex','NumColumns',2);

end
