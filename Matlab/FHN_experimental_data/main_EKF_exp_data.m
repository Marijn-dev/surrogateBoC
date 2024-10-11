clear all; clc; close all;
plot_settings
global R nx nth nz ny Qz pm

defaultColors = get(gca, 'ColorOrder');
customColors = defaultColors([1,2,3,4,5,6], :);

%% Initialise parameters
nx = 2; nth = 3; nz = 5; ny=1;
fun_parameters()
fun_overview_print_h5file(pm.loading.measurement, 'exp_data.txt')

%% Load and pre-process the experimental data
%load all electrode data
[t_data,y_raw,chanID,I,I_stim,t_stim] = fun_loadRawData(pm.loading.measurement);
y_data = fun_FilterData(y_raw,false); %filter the data

t_data = t_data*10^3; %scale to [ms]
t_stim = t_stim*10^3; %scale to [ms]
y_data = y_data*10^-3; %scale to [mV]
dt = pm.st*10^3;  %sample time in [ms]
pm.stim.duration = pm.stim.duration*10^3; %scale to [ms]

%% extract data of desired electrode
chanIDselect = {'G5'};  %{pm.stim.channelID};  {'G5'}; 
I = fun_SelectChanData(I,{pm.stim.channelID},chanIDselect);
y_data = fun_SelectChanData(y_data,chanID,chanIDselect);

%% Set up the EKF settings
init_guess = [0; 1; log(0.2); log(1.5); log(1)]; %[vh wh a'h b'h tau'h]
P_init = eye(nz); 
init_ODE = [init_guess; P_init(:)]; %[v_h w_h a_h b_h tau_h p11 p12...]

% Process and measurement noise covariance matrix
Qz = 0.002*eye(nz);
% Qz(1,1)=0.001;
% Qz(2,2)=0.001;
R = 0.01;

%% Extended Kalman Filter
h_z_log = zeros(length(t_data),nth+nx);
h_y_log = zeros(length(t_data),1);

% end_ind = 100; 
start_ind = int32((t_stim(1)-10)/dt);
end_ind = int32((t_stim(1)+50)/dt);
% end_ind = int32((t_stim(2)+100)/dt);

for i = start_ind:end_ind %length(t_data)  
    fprintf('at time: %d stop time: %d\n', t_data(i-start_ind+1), t_data(end_ind-start_ind));
    
    % Prediction step
    [t, q] = ode23(@(t,q) FHN_extended_discrete_data(t,q,t_data(i),t_stim,pm), [0 dt], init_ODE);
    h_z = q(end,1:5)';
    Parray = q(end,6:end);
    P = reshape(Parray, [nz, nz]);

    %%%%%%%% Set the output equation
    H = [(-1+h_z(1).^2) 1 0 0 0];
    h_y = -h_z(1) + ((1/3)*(h_z(1)^3)) + h_z(2) - funInput(t_data(i)+dt,t_stim,pm);
    %%%%%%%%
    
    % Update step
    K_k = P*H'/(H*P*H'+R);
    h_z_new = h_z + K_k*(y_data(i)-h_y);
    P_new = (eye(nz)-K_k*H)*P;
    
    init_ODE = [h_z_new; P_new(:)];
    
    h_z_log(i,:) = h_z_new;
    h_y_log(i) = h_y;
end

h_x = h_z_log(:,1:2);
h_th = exp(h_z_log(:,3:5));

%% Plot estimations
close all
fprintf('The EKF parameters: \n a=%4.4f, b=%4.4f, tau=%4.8f \n',h_th(end_ind,1),h_th(end_ind,2),h_th(end_ind,3))

t_plot = t_data(start_ind:end_ind)-t_data(start_ind);

I_plot = funInput(t_data(start_ind:end_ind),t_stim,pm);
figure()
subplot(3,1,1)
% plot(t_data(start_ind:end_ind), I(start_ind:end_ind),'Color',customColors(1,:),'LineWidth',1.2);
plot(t_plot, I_plot,'Color',customColors(1,:),'LineWidth',1.2);
ylabel('Input ($\mu A$)')
% xlabel('Time (s)')
xlim([t_plot(1) t_plot(end)])

subplot(3,1,[2 3])
plot(t_plot, y_data(start_ind:end_ind),'Color',customColors(3,:),'LineWidth',1.5); %C5BBFF 4DBEEE 00E874
hold on;
plot(t_plot, h_y_log(start_ind:end_ind),'Color',customColors(2,:))
ylabel('Measurement (mV)')
ylim([-0.5 0.5])
xlim([t_plot(1) t_plot(end)])
legend('$y_k$', '$h(\hat{x}_k^-, I_k)$','Location','best','Orientation','horizontal');
xlabel('Time (ms)')
v = get(gca,'Position');
set(gca,'Position',[v(1) v(2)*1.5 v(3) v(4)-v(2)*0.5])

if pm.save_results
    saveas(gcf, fullfile(pm.figure_folder, 'FHN_exp_data_output.eps'), 'epsc');
end

%%%%% plot of the state estimates
figure()
subplot(2,1,1)
plot(t_plot, h_x(start_ind:end_ind,1),'Color',customColors(4,:));
hold on 
plot(t_plot, h_x(start_ind:end_ind,2),'Color',customColors(5,:));
% ylim([-0.7 0.7])
% xlabel('Time (ms)')
ylabel('FHN States (mV/-)')
legend('$\hat{x}_1(t)$', '$\hat{x}_2(t)$','Location','best','Orientation','vertical');
xlim([t_plot(1) t_plot(end)])

subplot(2,1,2)
plot(t_plot, h_th(start_ind:end_ind,1),'Color',customColors(4,:));
hold on 
plot(t_plot, h_th(start_ind:end_ind,2),'Color',customColors(5,:));
plot(t_plot, h_th(start_ind:end_ind,3),'Color',customColors(6,:));
set(gca, 'YScale', 'log');
ylabel('FHN Parameters (-)')
legend('$\hat{x}_3(t)$','$\hat{x}_4(t)$','$\hat{x}_5(t)$','Location','northeast','Orientation','horizontal');
xlim([t_plot(1) t_plot(end)])
xlabel('Time (ms)')

if pm.save_results
    saveas(gcf, fullfile(pm.figure_folder, 'FHN_exp_data_states.eps'), 'epsc');
end
