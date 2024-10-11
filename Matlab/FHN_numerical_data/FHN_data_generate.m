function xdot = FHN_data_generate(t,x)
global th_true

%% Input
I = funInput(t);

%% calcualte the update+predict step
xdot = funf(x,I,th_true);

end