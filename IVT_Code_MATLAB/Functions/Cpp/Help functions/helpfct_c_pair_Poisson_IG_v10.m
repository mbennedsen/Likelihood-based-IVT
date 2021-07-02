function [loglik,grad] = helpfct_c_pair_Poisson_IG_v10(y,par,Hvec,H,n,dt)

output = c_pair_grad_Poisson_IG_v10(y,par,Hvec,H,n,dt);

loglik = -1*output(1);
grad   = -1*exp(par).*output(2:4);
%grad   = -1*output(2:4);