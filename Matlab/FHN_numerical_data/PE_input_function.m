%% solve the function

%set up the variables
syms a b g D I c t

v = c;
% v = sin(c*t)

%% Set up the function
term1 = (D^2+g*D+a)/(D+g);
term2 = b/(D+g);

func = term1*v+v^3/3-v+term2-I;

S = solve(func==0,v);