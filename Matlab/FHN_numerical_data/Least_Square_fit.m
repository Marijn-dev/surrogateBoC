clear all; close all; clc
plot_settings

%% Initalise parameters for the test data
Fs = 50; %[kHz]
tmax = 1000; %[ms]
t_span = 0:1/Fs:tmax; %[ms]
v0 = [-1.5; -3/8]; %[muV]
th_true = [0.7; 0.8; 0.08]; %[a b tau]

%% Make the input (comment out the type of input you want)
I = getI(t_span,Fs,100,2);                        %random pulses input
% I = randn(1, length(t_span));                     %White noise input
% I = 1*ones(1,length(t_span));                       %consistent input
% freq = linspace(0.0002, 0.0015, length(t_span));
% I = sin(2 * pi * freq .* t_span);                 %sinusoidal input
% I = max(I,0);                                     %Neglegt neg part

%% Calculate the trajectory
[t,x] = ode45(@(t,V)FHN(t,V,th_true,I,t_span),t_span,v0);

%% Get the Frequency data
% Perform Fourier analysis
L = length(t);
f = Fs*(0:(L/2))/L;
f_pos_neg = Fs/L*(-L/2:L/2-1);

omega_i = 2*pi*1i*f';
omega_i_pos_neg = 2*pi*f_pos_neg';

for i=1:size(x,2)
X(:,i) = fft(x(:,i),L);
P2(:,i) = abs(X(:,i)/L);
P1(:,i) = P2(1:L/2+1,i);
P1(2:end-1,i) = 2*P1(2:end-1,i);

X1(:,i) = X(1:L/2+1,i);
X1(2:end-1,i) = 2*X1(2:end-1,i);
end

V_fft = P1(:,1);
W_fft = P1(:,2);

%% Plot the system
figure(); 
subplot(1,3,1)
plot(t_span,I);
xlabel('Time (ms)'), ylabel('Input current ($mA$)')

subplot(1,3,2)
hold 'on'
plot(t, x(:,1), 'r', 'DisplayName','V')
plot(t, x(:,2), 'b', 'DisplayName','W')
xlabel('Time (ms)'), ylabel('Voltage ($\mu$V)')
legend('Location','Best')

subplot(1,3,3)
hold 'on'
plot(f, V_fft, 'r', 'DisplayName','$|fft(V)|$')
plot(f, W_fft, 'b', 'DisplayName','$|fft(W)|$')
xlabel('Frequency (kHz)'), ylabel('Magnitude FFT (dB)')
legend('Location','Best')
xlim([0 0.8])

%% get theta with COMPLEX data
% Calculate with entire frequency spectrum
% Y = omega_i_pos_neg.*X(:,2);
% H = [X(:,1) ones(size(X(:,1))) -1.*X(:,2)];

% Calculate with only positive frequency spectrum
Y = omega_i.*X1(:,2);
H = [X1(:,1) ones(size(X1(:,1))) -1.*X1(:,2)];

theta = inv(H'*H)*H'*Y; %[tau c d]

a_LS = real(theta(2))./real(theta(1));
b_LS = real(theta(3))./real(theta(1));
tau_LS = real(theta(1));
theta_LS = [a_LS b_LS tau_LS];

fprintf('The LS parameters: a=%4.2f, b=%4.2f, tau=%4.2f \n',theta_LS(1),theta_LS(2),theta_LS(3))
fprintf('And true parameters: a=%4.2f, b=%4.2f, tau=%4.2f \n',th_true(1),th_true(2),th_true(3))

return

%% Compare the trajectories
theta_LS = [real(a_LS) real(b_LS) abs(real(tau_LS))];
[t_LS,x_LS] = ode45(@(t,V)FHN(t,V,theta_LS,I,t_span),t_span,v0);

%plot the data as well as the fitted function
figure() 
subplot(1,2,1)
hold 'on'
plot(t, x(:,1), 'r--', 'DisplayName','$V_{true}$')
plot(t_LS, x_LS(:,1), 'r', 'DisplayName','$V_{LS}$')
xlabel('Time (ms)'), ylabel('Voltage ($\mu$V)')
legend('Location','Best')
xlim([0 0.2*t(end)])

subplot(1,2,2)
hold 'on'
plot(t, x(:,2), 'b--', 'DisplayName','$W_{true}$')
plot(t_LS, x_LS(:,2), 'b', 'DisplayName','$W_{LS}$')
xlabel('Time (ms)'), ylabel('Voltage ($\mu$V)')
legend('Location','Best')
xlim([0 0.2*t(end)])

%% FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vdot = FHN(t,V,th,I,t_span)
a = th(1);
b = th(2);
tau = th(3);

%Take the input value closest to this time stamp
[~,indx] = min(abs(t_span-t));
Input = I(indx);

vdot(1) = V(1)-((V(1).^3)./3)-V(2)+Input;
vdot(2) = tau*(V(1)+a-b*V(2));
vdot = vdot';
end

function I = getI(t_span,Fs,N,W)
% Generate random moments for biphasic pulses
pulse_moments = sort(rand(1, N) * t_span(end));

% Create signal with biphasic pulses
I = zeros(size(t_span));
for i = 1:N
    A = rand(1)*2;
    pulse_start = pulse_moments(i);
    pulse_shape = A * [ones(1, 2*round(W*Fs))]; %unidirectional
    
    % Insert pulse into the signal
    pulse_indices = round(pulse_start * Fs) + (1:length(pulse_shape));
    I(pulse_indices) = pulse_shape;
end
I = I(1:length(t_span));
end