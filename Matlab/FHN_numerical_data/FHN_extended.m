function qdot = FHN_extended(t,q)
global th_true R Qz nz
t
%% Extract the different states
%q = [v w v_h w_h a_h b_h tau_h p11 p12...]
x = q(1:2);
h_x = q(3:4);
h_th = q(5:7);
Parray = q(8:end);
P = reshape(Parray, [nz, nz]);

%% Input
I = funInput(t);

%% Extract the measurement
% y = x(1) + sqrt(R)*randn(1, 1);
% y = x(1);
% yhat = h_x(1);

y = -x(1) + ((1/3)*(x(1).^3))+ x(2) - I;
yhat = -h_x(1) + ((1/3)*(h_x(1).^3))+ h_x(2)- I;

%% Calculate F, K and H matrix
% H = [1 0 0 0 0];
H = [(-1+h_x(1).^2) 1 0 0 0];

% K = P*H'/R;
K = P*H'/(H*P*H'+R);
F =[1-h_x(1)^2 -1 0 0 0; 
    h_th(3) -h_th(2)*h_th(3) h_th(3) -h_x(2)*h_th(3) (h_x(1)+h_th(1)-h_x(2)*h_th(2));
    0 0 0 0 0;
    0 0 0 0 0;
    0 0 0 0 0;];

%% calcualte the update+predict step
xdot = funf(x,I,th_true); %+ sqrt(Qx)*randn(2, 1); 

h_xdot_update = funf(h_x,I,h_th);
h_thdot_update = zeros(length(h_th),1);
h_zdot = [h_xdot_update; h_thdot_update] + K*(y-yhat);

Pdot = F*P+P*F'-K*H*P+Qz;

qdot = [xdot; h_zdot; Pdot(:)];

end