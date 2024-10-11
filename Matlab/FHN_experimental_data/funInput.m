function I = funInput(t,t_stim,pm)

%% From exp, t_stim
I=0;
% for i=t_stim
%     I = I + pm.stim.amplitude*heaviside(t - i)- pm.stim.amplitude*heaviside(t - (i+pm.stim.duration));
% end
I = I + pm.stim.amplitude*heaviside(t - t_stim(1))- pm.stim.amplitude*heaviside(t - (t_stim(1)+pm.stim.duration));
I = I + pm.stim.amplitude*heaviside(t - t_stim(2))- pm.stim.amplitude*heaviside(t - (t_stim(2)+pm.stim.duration));


end

