function qdot = FHN_extended_discrete_data(t,q,t_input)
global Qz nz

%% Extract the different states
%q = [v_h w_h a_h b_h tau_h p11 p12...]
h_x = q(1:2);
% h_th = q(3:5);
h_th = exp(q(3:5));

Parray = q(6:end);
P = reshape(Parray, [nz, nz]);

%% Input
I = funInput(t_input+t);

%% Calculate F, K and H matrix
% F =[1-h_x(1)^2 -1 0 0 0; 
    % h_th(3) -h_th(2)*h_th(3) h_th(3) -h_x(2)*h_th(3) (h_x(1)+h_th(1)-h_x(2)*h_th(2));
    % 0 0 0 0 0;
    % 0 0 0 0 0;
    % 0 0 0 0 0;];
a_func = h_th(1);
b_func = h_th(2);
tau_func = h_th(3);

F =[1-h_x(1)^2, -1, 0, 0, 0; 
    tau_func, -b_func*tau_func, a_func*tau_func, -h_x(2)*b_func*tau_func, tau_func*(h_x(1)+a_func-h_x(2)*b_func);
    0, 0, 0, 0, 0;
    0, 0, 0, 0, 0;
    0, 0, 0, 0, 0;];
%% calcualte the update+predict step
h_xdot_update = funf(h_x,I,h_th);
h_thdot_update = zeros(length(h_th),1);
h_zdot = [h_xdot_update; h_thdot_update];

Pdot = F*P+P*F'+Qz;

qdot = [h_zdot; Pdot(:)];

end