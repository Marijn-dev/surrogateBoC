clear all; clc; close all;

% [0.7; 0.8; 0.08]; %[a b tau]

% Define the nonlinear 2nd order ODE
ode = @(t, x) [x(1)-((x(1).^3)./3)-x(2)+1;
               0.08*(x(1)+0.7-0.8*x(2))];

% Set initial state and covariance
initial_state_true = [-1.5; -3/8];
initial_state_guess = [-1.5; 2];

initial_covariance = eye(2);

% Process noise covariance matrix
Q = 0.01*eye(2);

% Measurement noise covariance matrix
R = 0.1;

% Time parameters
T = 100;   % Total simulation time
dt = 0.1;
time_span = [0:dt:T];

% Solve the ODE using ode45
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
[t, true_states] = ode45(ode, time_span, initial_state_true, options);

% Simulate noisy measurements
measurements = true_states(:, 1) + sqrt(R)*randn(size(true_states, 1), 1);

% Extended Kalman Filter
estimated_states = zeros(size(true_states));
predicted_states = zeros(size(true_states));
covariance = initial_covariance;

for i = 1:length(t)
    % Prediction step
    if i > 1
%         F = [1-estimated_states(i-1,1).^2 -1; 0.08 0.08*0.8];
        F = eye(2)+dt*[1-estimated_states(i-1,1).^2 -1; 0.08 0.08*0.8];
        
%         t_i = t(i);
%         predicted_states(i,:) = predicted_states(i-1,:) + dt*ode(t_i, estimated_states(i-1,:)')';
        t_span = [t(i-1), t(i)];
        [~, predicted_states_temp] = ode45(ode, t_span, estimated_states(i-1,:)', options);
        predicted_states(i,:) = predicted_states_temp(end,:);
        
        covariance = F * covariance * F' + Q;
    else
        predicted_states(i,:) = initial_state_guess';
    end
    
    % Update step
    H = [1 0];
    K = covariance * H' / (H * covariance * H' + R);
    estimated_states(i,:) = predicted_states(i,:) + transpose(K * (measurements(i) - predicted_states(i,1)));
    covariance = (eye(2) - K * H) * covariance;
end

% Plot results
figure;

subplot(2,1,1);
plot(t, true_states(:,1), 'b', 'LineWidth', 2);
hold on;
plot(t, measurements, 'ro', 'MarkerSize', 8);
plot(t, estimated_states(:,1), 'g--', 'LineWidth', 2);
legend('True State', 'Measurements', 'EKF Estimate');
title('Extended Kalman Filter for Nonlinear 2nd Order ODE - Position');

subplot(2,1,2);
plot(t, true_states(:,2), 'b', 'LineWidth', 2);
hold on;
plot(t, estimated_states(:,2), 'g--', 'LineWidth', 2);
legend('True Velocity', 'EKF Estimate');
title('Extended Kalman Filter for Nonlinear 2nd Order ODE - Velocity');
