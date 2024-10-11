function xdot = funf(x,u,theta)
%FHN state equations as a function of the parameters, states and inputs. 

a=theta(1); b=theta(2); tau=theta(3);
v=x(1); w=x(2);
I=u;

xdot(1)=v-((v.^3)./3)-w+I;
xdot(2)=tau*(v+a-b*w);
xdot=xdot';

end

