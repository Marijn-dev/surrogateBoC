clear all; clc; close all;
plot_settings

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,3,2,4,5,6], :);
%% Simulation parameters
global th_true R nx nth nz ny Qz I_HS Rv t_pulse A_pulse dt_pulse

nx = 2; nth = 3; nz = 5; ny=1;

%Initialise the states, estimates (covariance), and paramaeters
th_true = [0.7; 0.8; 0.08]; %[a b tau]
[Rv,Rw] = fun_restState(th_true(1),th_true(2));
true_restState = [Rv; Rw]; %true_init_state = [-1.5; -3/8]; %[v w]

% init_guess = [Rv; Rw+2; 0.2; 1.5; 0.01]; %[vh wh ah bh tauh]
init_guess = [Rv; Rw+2; log(0.2); log(1.5); log(1)]; %[vh wh a'h b'h tau'h]

P_init = eye(nz); 
init_ODE = [init_guess; P_init(:)]; %[v_h w_h a_h b_h tau_h p11 p12...]

% Process and measurement noise covariance matrix
Qz = 0.001*eye(nz);
Qz(1,1)=0;
Qz(2,2)=0;
R = 0.01;

% Time parameters
T = 300;   % Total simulation time
dt = 0.05;
time_span = [0:dt:T];

%Input design parameters
I_HS = 50;
t_pulse = 1;
dt_pulse = 20;
A_pulse = 1;

%% Simulate the measured data
[t_data, q_data] = ode23(@(t,q) FHN_data_generate(t,q), time_span, true_restState);

% Add noise to the measurements
I_sol = funInput(t_data);
y = -q_data(:,1) + ((1/3)*(q_data(:,1).^3)) + q_data(:,2) - I_sol;
% y = q_data(:,1);

y_noise = y + sqrt(R)*randn(size(q_data, 1), 1);

%% Extended Kalman Filter
h_z_log = zeros(length(t_data),nth+nx);
h_y_log = zeros(length(t_data),1);

for i = 1:length(t_data)   
    t_data(i)
    % Prediction step
    [t, q] = ode23(@(t,q) FHN_extended_discrete_data(t,q,t_data(i)), [0 dt], init_ODE);
    h_z = q(end,1:5)';
    Parray = q(end,6:end);
    P = reshape(Parray, [nz, nz]);

    %%%%%%%% Set the output equation
    H = [(-1+h_z(1).^2) 1 0 0 0];
    h_y = -h_z(1) + ((1/3)*(h_z(1)^3)) + h_z(2) - funInput(t_data(i)+dt);
    
    %%%%%%%%
    
    % Update step
    K_k = P*H'/(H*P*H'+R);
    h_z_new = h_z + K_k*(y_noise(i)-h_y);
    P_new = (eye(nz)-K_k*H)*P;
    
    init_ODE = [h_z_new; P_new(:)];
    
    h_z_log(i,:) = h_z_new;
    h_y_log(i) = h_y;
end

h_x = h_z_log(:,1:2);
% h_th = h_z_log(:,3:5);
h_th = exp(h_z_log(:,3:5));

%% Parameter estimates
fprintf('The EKF parameters: a=%4.4f, b=%4.4f, tau=%4.4f \n',h_th(end,1),h_th(end,2),h_th(end,3))
fprintf('The true parameters: a=%4.4f, b=%4.4f, tau=%4.4f \n',th_true(1),th_true(2),th_true(3))

errors = h_th- [th_true(1)*ones(length(t_data),1) th_true(2)*ones(length(t_data),1) th_true(3)*ones(length(t_data),1)];
avg_error =  mean(errors,2);
error_perc = errors./th_true';
init_error_perc = error_perc(1,:)*100;

x = q_data;

fprintf('Initial error a: %4.2f , b: %4.2f, tau %4.2f  \n', init_error_perc(1),init_error_perc(2),init_error_perc(3))
fprintf('mean estimation error: %12.8f  \n', mean(abs(errors(end,:))))

%% Figure V2

figure_folder = fullfile('D:\OneDrive - TU Eindhoven\Master 3\neurostimulus\Matlab_FHNmodel\', 'Figures_thesis');

close all
% figure()
% 
% yyaxis left
% plot(t_data, y_noise,'Color',customColors(2,:),'LineWidth',1.5); %C5BBFF 4DBEEE 00E874
% hold on;
% plot(t_data, h_y_log,'--','Color',customColors(3,:))
% % plot(t_data, funInput(t_data),'Color',customColors(1,:),'LineWidth',1.2);
% set(gca, 'YColor', 'black'); 
% ylabel('Measurement (mV)')
% ylim([-2.5 2.5])
% 
% yyaxis right
% plot(t_data, funInput(t_data),'Color',customColors(1,:),'LineWidth',1.2);
% ylabel('Input stimulation ($\mu A$)')
% % set(gca, 'YColor', customColors(1,:)); % Set color for left y-axis tick marks
% set(gca, 'YColor', 'black'); 
% legend('$y_k$', '$h(\hat{x}_k^-, I_k)$','$I_k$','Location','best','Orientation','horizontal');
% ylim([-3.5 3.5])
% xlabel('Time (ms)')
% saveas(gcf, fullfile(figure_folder, 'FHN_output.eps'), 'epsc');


figure()
subplot(3,1,1)
plot(t_data, funInput(t_data),'Color',customColors(1,:),'LineWidth',1.2);
ylabel('Input ($\mu A$)')
% xlabel('Time (s)')
ylim([0 1.2])

subplot(3,1,[2 3])
plot(t_data, y_noise,'Color',customColors(2,:),'LineWidth',1.5); %C5BBFF 4DBEEE 00E874
hold on;
plot(t_data, h_y_log,'--','Color',customColors(3,:))
ylabel('Measurement (mV)')
% ylim([-2.5 2.5])
legend('$y_k$', '$h(\hat{x}_k^-, I_k)$','Location','best','Orientation','horizontal');
xlabel('Time (ms)')
v = get(gca,'Position');
set(gca,'Position',[v(1) v(2)*1.5 v(3) v(4)-v(2)*0.5])

% saveas(gcf, fullfile(figure_folder, 'FHN_subplot_trajectories.eps'), 'epsc');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure()
subplot(2,1,2)
yline(th_true(1),'LineWidth',1.5,'Color',customColors(4,:));
hold on;
plot(t_data, h_th(:,1),'--','LineWidth',2,'Color',customColors(4,:)); 
yline(th_true(2),'LineWidth',1.5,'Color',customColors(5,:));
plot(t_data, h_th(:,2),'--','LineWidth',2,'Color',customColors(5,:)); 
yline(th_true(3),'LineWidth',1.5,'Color',customColors(6,:));
plot(t_data, h_th(:,3),'--','LineWidth',2,'Color',customColors(6,:)); 
ylabel('FHN Parameters (-)')
xlabel('Time (ms)')
legend('$x_3(t)$', '$\hat{x}_3(t)$','$x_4(t)$', '$\hat{x}_4(t)$','$x_5(t)$', '$\hat{x}_5(t)$','Location','best','NumColumns',3); %,'Orientation','horizontal');
set(gca, 'YScale', 'log')
ylim([0.01 10])

% saveas(gcf, fullfile(figure_folder, 'FHN_param_est.eps'), 'epsc');

subplot(2,1,1)
plot(t_data, x(:,1),'LineWidth',1.5,'Color',customColors(4,:));
hold on;
plot(t_data, h_x(:,1),'--','LineWidth',2,'Color',customColors(4,:)); 
plot(t_data, x(:,2),'LineWidth',1.5,'Color',customColors(5,:));
plot(t_data, h_x(:,2),'--','LineWidth',2,'Color',customColors(5,:)); 
ylabel('FHN States (mV/-)')
% xlabel('Time (ms)')
legend('$x_1(t)$', '$\hat{x}_1(t)$','$x_2(t)$', '$\hat{x}_2(t)$','Location','best','Orientation','horizontal'); %,'NumColumns',2
saveas(gcf, fullfile(figure_folder, 'FHN_state_est.eps'), 'epsc');
ylim([-2.2 3])




%% figure for in thesis
% close all
% % screenSize = get(0, 'ScreenSize');
% % fig = figure('Position', [1, 1, 0.5*screenSize(3), 0.6*screenSize(4)]);
% figure()
% subplot(4,1,1)
% plot(t_data, funInput(t_data),'Color',customColors(1,:),'LineWidth',1.2);
% ylabel('I ($mA$)')
% 
% subplot(4,1,2)
% plot(t_data, y_noise,'Color',customColors(2,:),'LineWidth',1.5); %C5BBFF 4DBEEE 00E874
% hold on;
% plot(t_data, h_y_log,'--','Color',customColors(3,:))
% ylabel('Measurement ($\mu$V)')
% legend('Data','Model');
% legend('$y_k$', '$h(\hat{x}_k^-, I_k$','Location','best');
% 
% subplot(4,1,3)
% plot(t_data, x(:,1),'Color',customColors(2,:),'LineWidth',1.5);
% hold on;
% ylabel('v ($\mu$V)')
% plot(t_data, h_x(:,1),'--','Color',customColors(3,:),'MarkerSize', 8); %, 'r-' "#7E2F8E"
% 
% subplot(4,1,4)
% plot(t_data, x(:,2),'Color',customColors(2,:),'LineWidth',1.5);
% hold on
% plot(t_data, h_x(:,2),'--','Color',customColors(3,:),'MarkerSize', 8); %, 'r-' "#7E2F8E"
% xlabel('Time (ms)'), ylabel('w (-)')
% legend('$\bar{x}(t)$', '$\hat{x}(t)$','Location','best');
% ylim([-1.3 2])	
% 
% % plot the parameter estimation
% figure()
% subplot(3,1,1);
% yline(th_true(1),'Color',customColors(2,:),'LineWidth',1.5);
% hold on;
% ylabel('$\bar{x}_3(t)$ / $a$ (-)')
% plot(t_data, h_th(:,1),'--','Color',customColors(3,:),'LineWidth',2);
% 
% subplot(3,1,2);
% yline(th_true(2),'Color',customColors(2,:),'LineWidth',1.5);
% hold on;
% plot(t_data, h_th(:,2),'--','Color',customColors(3,:),'LineWidth',2);
% ylabel('$\bar{x}_4(t)$ / $b$ (-)')
% 
% subplot(3,1,3);
% yline(th_true(3),'Color',customColors(2,:),'LineWidth',1.5);
% hold on;
% plot(t_data, h_th(:,3),'--','Color',customColors(3,:),'LineWidth',2);
% xlabel('Time (ms)'), ylabel('$\bar{x}_5(t)$ / $\tau$ (-)')
% legend('$\bar{x}(t)$', '$\hat{x}(t)$','Location','best');

%%
% close all
% % screenSize = get(0, 'ScreenSize');
% 
% % fig = figure('Position', [1, 1, 0.5*screenSize(3), 0.6*screenSize(4)]);
% figure()
% subplot(4,1,1)
% plot(t_data, funInput(t_data),'k-','LineWidth',1.2);
% % title('Stimulation of neuron in front of the tunnel');
% ylabel('I ($mA$)')
% 
% subplot(4,1,2)
% plot(t_data, y_noise,'Color', "#C5BBFF", 'MarkerSize', 8); %C5BBFF 4DBEEE 00E874
% hold on;
% plot(t_data, h_y_log,'b--')
% ylabel('y ($\mu$V)')
% legend('Data','Model');
% % title('Extracellular measurement');
% 
% subplot(4,1,3)
% plot(t_data, x(:,1),'g');
% hold on;
% ylabel('v ($\mu$V)')
% plot(t_data, h_x(:,1),'b--' ,'MarkerSize', 8); %, 'r-' "#7E2F8E"
% legend('True','Model');
% % title('Axon membrane potential');
% 
% subplot(4,1,4)
% plot(t_data, x(:,2),'g');
% hold on
% plot(t_data, h_x(:,2),'b--' ,'MarkerSize', 8); %, 'r-' "#7E2F8E"
% % title('Axon relaxation state');
% xlabel('Time (ms)'), ylabel('w (-)')
% ylim([-1.3 2])	
% 
% 
% % plot the parameter estimation
% figure()
% subplot(3,1,1);
% yline(th_true(1),'Color', "#EDB120",'LineWidth',1.5);
% hold on;
% ylabel('$a$ (-)')
% plot(t_data, h_th(:,1),'Color', "#D95319");
% % title('Estimated parameter: $a$');
% 
% subplot(3,1,2);
% yline(th_true(2),'Color',"#EDB120",'LineWidth',1.5);
% hold on;
% plot(t_data, h_th(:,2),'Color', "#D95319");
% % legend('True', 'EKF');
% ylabel('$b$ (-)')
% % title('Estimated parameter: $b$');
% 
% subplot(3,1,3);
% yline(th_true(3),'Color',"#EDB120",'LineWidth',1.5);
% hold on;
% plot(t_data, h_th(:,3),'Color', "#D95319");
% xlabel('Time (ms)'), ylabel('$\tau$ (-)')
% % title('Estimated parameter: $\tau$');
% legend('True', 'EKF','Location','best');
% 
% 
