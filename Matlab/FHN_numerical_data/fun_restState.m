function [Sv,Sw] = fun_restState(a,b)
syms vsym

Sv = double(vpasolve(vsym-vsym^3/3-(vsym+a)/b == 0,vsym,[-inf inf]));
Sw = (Sv+a)/b;

end

