function I = funInput(t)
global I_HS T t_pulse dt_pulse A_pulse

%% Step
I = 0.5*heaviside(t - 0.5*I_HS)-0.5*heaviside(t - (0.5*I_HS+25));
I = I+heaviside(t - 2*I_HS)-heaviside(t - 4*I_HS);
I = I+0.2*(heaviside(t - 4.5*I_HS)-heaviside(t - (4.8*I_HS)));
I = I+0.4*(heaviside(t - 5.5*I_HS)-heaviside(t - (5.8*I_HS)));
I = I+heaviside(t - 7*I_HS)-heaviside(t - 9*I_HS);
% I = I+heaviside(t - 10*I_HS);
%% Block
% N_pulses = T/dt_pulse;

% I = A_pulse*heaviside(t - I_HS)-A_pulse*heaviside(t - (I_HS+t_pulse));
% for i=1:N_pulses
%     x1 = i*dt_pulse+I_HS;
%     x2 = x1+t_pulse;
%     I = I+A_pulse*heaviside(t - x1)-A_pulse*heaviside(t - x2);
% end

%% Constant
% I = wgn(size(t,1),size(t,2),2); %rand(size(t));
% I = A_pulse*ones(size(t));

% I = A_pulse*heaviside(t - I_HS);
% I = 0.25*sin(t);
% I = rand(1,1);
% I = heaviside(t - I_HS);
% I = ones(size(t));

end

