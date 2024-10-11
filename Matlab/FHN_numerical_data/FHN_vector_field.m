clear all; clc; close all;
plot_settings

%% Parameters
tau = 0.08; a = 0.7; b = 0.8;
I = 0;

%%
syms vsym

Sv = vpasolve(vsym-vsym^3/3-(vsym+a)/b == 0,vsym,[-3 3]);
Sw = (Sv+a)/b;

disp('Resting state (I=0):');
fprintf('v_C = %2.2f and w_C = %2.2f \n', Sv,Sw);

%% Plot results
[v,w] = meshgrid(-2.5:0.15:2.5, -1.5:0.15:1.5);
dv = v-((v.^3)./3)-w+I;
dw = tau*(v+a-b*w);

vvec = -2.5:0.1:2.5;
wcline1 = vvec-((vvec.^3)./3)+I;
wcline2 = (vvec+a)./b;

colors = lines(5);

%%
fig = figure();
hold on
plot(vvec, wcline1,'LineWidth',1.2, 'color', colors(2,:), 'DisplayName', 'v-ncline')
plot(vvec, wcline2,'LineWidth',1.2, 'color', colors(3,:), 'DisplayName', 'w-ncline')
plot(Sv, Sw, '.', 'Markersize', 25, 'color', colors(4,:), 'DisplayName', 'resting state');
quiver(v, w, 4*dv, 4*dw,'LineWidth',1.2, 'color', colors(1,:), 'HandleVisibility', 'off');

xlabel('Membrane potential (v)')
ylabel('Recovery variable (w)')
xlim([-2.5 2.5]); ylim([-1.5 1.5])
title('a=0.7, b=0.8, $\tau=0.08$ and I=0')
legend('Show')

saveas(fig,'Figures\FHN_vector_field_I0','epsc')

