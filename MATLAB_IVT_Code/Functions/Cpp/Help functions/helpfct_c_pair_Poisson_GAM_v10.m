function [loglik,grad] = helpfct_c_pair_Poisson_GAM_v10(y,par,Hvec,H,n,dt)

output = c_pair_grad_Poisson_GAM_v10(y,par,Hvec,H,n,dt);

loglik = -1*output(1);
grad   = -1*exp(par(1:3)).*output(2:4);
%          
% grad   = -1*[exp(par(1)).*output(2);
%              5*sigmoid(par(2))^2.*exp(-par(2)).*output(3);
%              exp(par(3)).*output(4)];
