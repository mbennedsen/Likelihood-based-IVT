function [model_num,IC] = modelselect_IVT(y,dt,K,B,N,options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Model selection for IVT process on equidistant grid. Six possible DGPs
% are considered: Poisson-Exp, Poisson-IG, Poisson-Gamma, NB-Exp, NB-IG, NB-Gamma.
%
% INPUT
% y          : Data (equidistant).
% dt:        : Time between (equidistant) observations.
% K          : Number of different pairwise observations to include in CL estimator.
% B          : Number of bootstrap replications used in calculation of covariance matrix. (Default = 500)
% N          : Number of observations simulated for every bootstrap replication. (Default = 500)
% options    : Options for numerical optimizer.
%
% OUTPUT
% model_num  : (3 x 1) vector denoting the selected models based on three criteria: Maximized composite likelihood (CL), CLAIC and CLBIC.
%              1: Poisson-Exp. 2: Poisson-IG. 3: Poisson-Gamma. 4: NB-Exp. 5: NB-IG. 6: NB-Gamma.
% IC         : Struct with Information Criteria values for each of the six IVT DGPs. 
%              IC.CL has maximized CL values; IC.CLAIC contains CLAIC values and IC.BIC contains CLBIC values.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% (c) Mikkel Bennedsen (2021)
%
% This code can be used, distributed, and changed freely. Please cite Bennedsen,
% Lunde, Shephard, and Veraart (2021): "Inference and forecasting for continuous 
% time integer-valued trawl processes and their use in financial economics".
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% init
Hvec = 1:K; % Lags used in CL estimator.
n = length(y);

if nargin<6
    options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'TolX',1e-15,'TolFun',1e-15,'MaxFunEvals',10000,'Display','off');
end

if nargin < 5
    N = 100;
end

if nargin < 4
    B = 100;
end

%% Checks
if sum(abs(mod(y,1)))>0
    error('Input data must be integer-valued.');
end
if sum(y<0)
    error('Input data must be non-negative.');
end
if ~(mod(K,1)==0) || K<0.50
    error('Input K (number of different lags to use in MCL estimator) must be a positive integer.');
end
if ~(dt>0)
    error('Time between observations (dt) must be a positive number.');
end

%% Estimate + get std errs + get IC: Poisson+Exp
param0 = log([mean(y);1]);

%%% Estimate
loglik_temp = @(par)( helpfct_c_pair_Poisson_Exp_v10(y,par,Hvec,length(Hvec),n,dt) );
optFct = @(par)( loglik_temp(par) );

[p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);

nu_hat  = exp(p_tmp(1));
lam_hat = exp(p_tmp(2));

loglik = -1*fval;
th_hat = [nu_hat;lam_hat];

p_tmp2 = p_tmp;

if loglik == 0
    loglik = NaN;
end

%% Bootstrap std errs
boot_gdp_num = 1;
s_boot = nan(B,length(p_tmp2));
for iB = 1:B
    if iB == 1
        [x_b,t_b,bob] = simulate_IVT(th_hat,boot_gdp_num,N,dt);
    else
        x_b = simulate_IVT(th_hat,boot_gdp_num,N,dt,bob);
    end
    
    [tmp1,tmp2] = helpfct_c_pair_Poisson_Exp_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
    s_boot(iB,:) = -tmp2';
end

%% AVAR calcs
V = cov(s_boot/sqrt(N));
H = -1*hessian/n;

AIC = loglik + trace( V\H );
BIC = loglik + 0.5*log(n)*trace( V\H );

poisExp_LL = loglik;
poisExp_IC = AIC;
poisExp_BIC = BIC;


%% Estimate + get std errs: Poisson-IG
param0 = log([mean(y);1;1]);

%%% Estimate
loglik_temp = @(par)( helpfct_c_pair_Poisson_IG_v10(y,par,Hvec,length(Hvec),n,dt) );
optFct = @(par)( loglik_temp(par) );

[p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);

nu_hat  = exp(p_tmp(1));
del_hat = exp(p_tmp(2));
gam_hat = exp(p_tmp(3));

loglik = -1*fval;
th_hat = [nu_hat;del_hat;gam_hat];

p_tmp2 = p_tmp;

if loglik == 0
    loglik = NaN;
end
%% Bootstrap std errs
boot_gdp_num = 2;
s_boot = nan(B,length(p_tmp2));
for iB = 1:B
    if iB == 1
        [x_b,t_b,bob] = simulate_IVT(th_hat,boot_gdp_num,N,dt);
    else
        x_b = simulate_IVT(th_hat,boot_gdp_num,N,dt,bob);
    end
    
    [tmp1,tmp2] = helpfct_c_pair_Poisson_IG_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
    s_boot(iB,:) = -tmp2';
end

%% AVAR calcs
V = cov(s_boot/sqrt(N));
H = -1*hessian/n;

AIC = loglik + trace( V\H );
BIC = loglik + 0.5*log(n)*trace( V\H );

poisIG_LL = loglik;
poisIG_IC = AIC;
poisIG_BIC = BIC;

%% Poisson-Gamma
param0 = log([mean(y);1;1]);

%%% Estimate
loglik_temp = @(par)( helpfct_c_pair_Poisson_GAM_v10(y,par,Hvec,length(Hvec),n,dt) );
optFct = @(par)( loglik_temp(par) );

[p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);

nu_hat  = exp(p_tmp(1));
H_hat   = exp(p_tmp(2));
alp_hat = exp(p_tmp(3));

loglik = -1*fval;
th_hat = [nu_hat;H_hat;alp_hat];

p_tmp2 = p_tmp;

if loglik == 0
    loglik = NaN;
end
%% Bootstrap std errs
boot_gdp_num = 3;
s_boot = nan(B,length(p_tmp2));
for iB = 1:B
    if iB == 1
        [x_b,t_b,bob] = simulate_IVT(th_hat,boot_gdp_num,N,dt);
    else
        x_b = simulate_IVT(th_hat,boot_gdp_num,N,dt,bob);
    end
    
    [tmp1,tmp2] = helpfct_c_pair_Poisson_GAM_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
    s_boot(iB,:) = -tmp2';
end

%% AVAR calcs
b_val = max(1,2-H_hat);

V = cov(s_boot/N^(b_val/2));
H = -1*hessian/n;

AIC = loglik + trace( V\H );
BIC = loglik + 0.5*log(n)*trace( V\H );

poisGAM_LL = loglik;
poisGAM_IC = AIC;
poisGAM_BIC = BIC;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% NB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NB-Exp
param0 = [log(mean(y)),0,0];

%%% Estimate
loglik_temp = @(par)( helpfct_c_pair_NB_Exp_v10(y,par,Hvec,length(Hvec),n,dt) );
optFct = @(par)( loglik_temp(par) );

[p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);

m_hat = exp(p_tmp(1));
p_hat = sigmoid(p_tmp(2));
lam_hat = exp(p_tmp(3));

loglik = -1*fval;
th_hat = [m_hat;p_hat;lam_hat];

p_tmp2 = p_tmp;

if loglik == 0
    loglik = NaN;
end

%% Bootstrap std errs
boot_gdp_num = 4;
s_boot = nan(B,length(p_tmp2));
for iB = 1:B
    if iB == 1
        [x_b,t_b,bob] = simulate_IVT(th_hat,boot_gdp_num,N,dt);
    else
        x_b = simulate_IVT(th_hat,boot_gdp_num,N,dt,bob);
    end
    
    [tmp1,tmp2] = helpfct_c_pair_NB_Exp_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
    s_boot(iB,:) = -tmp2';
end

%% AVAR calcs
V = cov(s_boot/sqrt(N));
H = -1*hessian/n;

AIC = loglik + trace( V\H );
BIC = loglik + 0.5*log(n)*trace( V\H );

NB_Exp_LL  = loglik;
NB_Exp_IC  = AIC;
NB_Exp_BIC = BIC;


%% NB-IG
param0 = [log(mean(y));0;log(1);log(1)];
%%% Estimate
loglik_temp = @(par)( helpfct_c_pair_NB_IG_v10(y,par,Hvec,length(Hvec),n,dt) );
optFct = @(par)( loglik_temp(par) );

% [p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);
[p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);

m_hat = exp(p_tmp(1));
p_hat = sigmoid(p_tmp(2));
del_hat = exp(p_tmp(3));
gam_hat = exp(p_tmp(4));

loglik = -1*fval;
th_hat = [m_hat;p_hat;del_hat;gam_hat];

p_tmp2 = p_tmp;

if loglik == 0
    loglik = NaN;
end
%% Bootstrap std errs
boot_gdp_num = 5;
s_boot = nan(B,length(p_tmp2));
for iB = 1:B
    if iB == 1
        [x_b,t_b,bob] = simulate_IVT(th_hat,boot_gdp_num,N,dt);
    else
        x_b = simulate_IVT(th_hat,boot_gdp_num,N,dt,bob);
    end
    
    [tmp1,tmp2] = helpfct_c_pair_NB_IG_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
    s_boot(iB,:) = -tmp2';
end

%% AVAR calcs
V = cov(s_boot/sqrt(N));
H = -1*hessian/n;

AIC = loglik + trace( V\H );
BIC = loglik + 0.5*log(n)*trace( V\H );

NB_IG_LL  = loglik;
NB_IG_IC  = AIC;
NB_IG_BIC = BIC;

%% NB-Gamma
param0 = [log(mean(y));0;log(1);log(1)];

%%% Estimate
loglik_temp = @(par)( helpfct_c_pair_NB_GAM_v10(y,par,Hvec,length(Hvec),n,dt) );
optFct = @(par)( loglik_temp(par) );

% [p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);
[p_tmp,fval,exitflag,output,grad,hessian] = fminunc(optFct,param0,options);

m_hat    = exp(p_tmp(1));
p_hat    = sigmoid(p_tmp(2));
H_hat  = exp(p_tmp(3));
alp_hat  = exp(p_tmp(4));

loglik = -1*fval;
th_hat = [m_hat;p_hat;H_hat;alp_hat];

p_tmp2 = p_tmp;

if loglik == 0
    loglik = NaN;
end

%% Bootstrap std errs
boot_gdp_num = 6;
s_boot = nan(B,length(p_tmp2));
for iB = 1:B
    if iB == 1
        [x_b,t_b,bob] = simulate_IVT(th_hat,boot_gdp_num,N,dt);
    else
        x_b = simulate_IVT(th_hat,boot_gdp_num,N,dt,bob);
    end
    
    [tmp1,tmp2] = helpfct_c_pair_NB_GAM_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
    s_boot(iB,:) = -tmp2';
end

%% AVAR calcs
b_val = max(1,2-H_hat);

V = cov(s_boot/N^(b_val/2));
H = -1*hessian/n;

AIC = loglik + trace( V\H );
BIC = loglik + 0.5*log(n)*trace( V\H );

NB_GAM_LL  = loglik;
NB_GAM_IC  = AIC;
NB_GAM_BIC = BIC;

%% Record LL choice
tmptmp = [poisExp_LL,poisIG_LL,poisGAM_LL,NB_Exp_LL,NB_IG_LL,NB_GAM_LL];
[val,indx] = max(tmptmp);

model_num = indx;
IC.CL = tmptmp;

%% Record IC choice
tmptmp = [poisExp_IC,poisIG_IC,poisGAM_IC,NB_Exp_IC,NB_IG_IC,NB_GAM_IC];
[val,indx] = max(tmptmp);

model_num = [model_num;indx];
IC.CLAIC = tmptmp;
%% Record BIC choice
tmptmp = [poisExp_BIC,poisIG_BIC,poisGAM_BIC,NB_Exp_BIC,NB_IG_BIC,NB_GAM_BIC];
[val,indx] = max(tmptmp);

model_num = [model_num;indx];
IC.CLBIC = tmptmp;