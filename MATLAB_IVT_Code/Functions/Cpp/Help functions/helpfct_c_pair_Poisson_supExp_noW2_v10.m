function [loglik,grad] = helpfct_c_pair_Poisson_supExp_noW2_v10(y,par,Hvec,H,n,dt,w)

output = c_pair_grad_Poisson_supExp_noW2_v10(y,par,Hvec,H,n,dt,w);

loglik = -1*output(1);
grad   = -1*exp(par(1:3)).*output(2:4);
% grad   = -1*[exp(par(1)).*output(2);
%              4*sigmoid(par(2)).^2.*exp(-par(2)).*output(3); 
%              sigmoid(par(3)).^2.*exp(-par(3)).*output(4)];

% lam1 = 1+4*sigmoid(par(2));
% lam2 = sigmoid(par(3));
% 
% grad   = -1*[exp(par(1)).*output(2);
%              (1/(lam1-1) + 1/(4+1-lam1))^(-1)*output(3); 
%              lam2*(1-lam2)*output(4)];

