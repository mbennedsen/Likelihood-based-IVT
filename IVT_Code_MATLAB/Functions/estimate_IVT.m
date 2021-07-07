function [beta_hat,std_err,cl,aic,bic] = estimate_IVT(y,dt,K,dgp_num,param0,options,B,N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate IVT process observed on equidistant grid.
%
% INPUT
% y          : Data (equidistant).
% dt:        : Time between (equidistant) observations.
% K          : Number of different pairwise observations to include in CL estimator.
% dgp_num:   : IVT DGP. 1: Poisson-Exp. 2: Poisson-IG. 3: Poisson-Gamma. 4: NB-Exp. 5: NB-IG. 6: NB-Gamma.
% param0     : Starting values for parameters used in numerical optimization procedure.
% options    : Options for numerical optimizer.
% B          : Number of bootstrap replications used in calculation of standard errors. (Default = 500)
% N          : Number of observations simulated for every bootstrap replication. (Default = 500)
%
% OUTPUT
% beta_hat   : Parameter estimates via MCL.
%
% OPTIONAL OUTPUT
% std_err    : Standard errors of estimated MCL parameter vector.
% cl         : Maximized composite likelihood value.
% aic        : CLAIC value.
% bic        : CLBIC value.
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


%% Init
Hvec = 1:K; % Lags used in CL estimator.
n = length(y);

if nargout>1 % Include standard errors
    use_grad = 1;
    
    if nargin < 8
        N = 500; % Number of observations to simulate in each bootstrap replication
    end
    
    if nargin < 7
        B = 500; % Number of bootstrap replication.
    end
    
    if nargin<6 || isempty(options)
       options = optimoptions(@fminunc,'Algorithm','trust-region','SpecifyObjectiveGradient',true,'TolX',1e-15,'TolFun',1e-15,'MaxFunEvals',10000,'Display','off');
    end
    
  
else
    use_grad = 0;
    
    if nargin<7
        options = optimset('TolX',1e-15,'TolFun',1e-15);
        options = optimset(options,'MaxFunEvals',10000,'Display','off');
    end
end

%% Perform checks
if sum(abs(mod(y,1)))>0
    error('Input data must be integer-valued.');
end
if sum(y<0)
    error('Input data must be non-negative.');
end
if ~(mod(K,1)==0) || K<0.50
    error('Input K (number of different lags to use in MCL estimator) must be a positive integer.');
end
if ~(dgp_num == 1 || dgp_num == 2 || dgp_num == 3 || dgp_num == 4 || dgp_num == 5 || dgp_num == 6)
    error('Input DGP number must be in {1,2,3,4,5,6}.');    
end
if ~(dt>0)
    error('Time between observations (dt) must be a positive number.');
end

%% Poisson-Exp
if dgp_num == 1

    if nargin < 5 || isempty(param0)
        param0 = log([mean(y);1]);
    end

    %%% Estimate
    if use_grad == 1
        loglik_temp = @(par)( helpfct_c_pair_Poisson_Exp_v10(y,par,Hvec,length(Hvec),n,dt) );
    else
        loglik_temp = @(par)( -c_pair_Poisson_Exp_v10(y,par,Hvec,length(Hvec),n,dt) );
    end
    
    optFct = @(par)( loglik_temp(par) );

    if use_grad == 1
        [p_tmp,fval,~,~,~,hessian] = fminunc(optFct,param0,options);
    else
        p_tmp = fminunc(optFct,param0,options);
    end

    nu_hat  = exp(p_tmp(1));
    lam_hat = exp(p_tmp(2));
    beta_hat = [nu_hat;lam_hat];

    if nargout>1
        cl = -1*fval;

        p_tmp2 = p_tmp;

        if cl == 0
            cl = NaN;
        end

        %% Bootstrap std errs
        s_boot = nan(B,length(p_tmp2));
        for iB = 1:B
            if iB == 1
                [x_b,t_b,bo] = simulate_IVT(beta_hat,dgp_num,N,dt);
            else
                x_b = simulate_IVT(beta_hat,dgp_num,N,dt,bo);
            end

            [tmp1,tmp2] = helpfct_c_pair_Poisson_Exp_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
            s_boot(iB,:) = -tmp2';
        end

        %% AVAR calcs
        V = cov(s_boot/sqrt(N));
        H = -1*hessian/n;
        G_inv = (H\V)/H;

        std_err = [exp(p_tmp2(1));exp(p_tmp2(2))].*sqrt(diag(G_inv))/sqrt(n);


        if nargout>3
            aic = cl + trace( V\H );
            bic = cl + 0.5*log(n)*trace( V\H );
        end
    end
    

%% Poisson-IG
elseif dgp_num == 2

    if nargin < 5 || isempty(param0)
        param0 = log([mean(y);1;1]);
    end

    %%% Estimate
    if use_grad == 1
        loglik_temp = @(par)( helpfct_c_pair_Poisson_IG_v10(y,par,Hvec,length(Hvec),n,dt) );
    else
        loglik_temp = @(par)( -c_pair_Poisson_IG_v10(y,par,Hvec,length(Hvec),n,dt) );
    end
    
    optFct = @(par)( loglik_temp(par) );

    if use_grad == 1
        [p_tmp,fval,~,~,~,hessian] = fminunc(optFct,param0,options);
    else
        p_tmp = fminunc(optFct,param0,options);
    end

    nu_hat  = exp(p_tmp(1));
    del_hat = exp(p_tmp(2));
    gam_hat = exp(p_tmp(3));
    beta_hat = [nu_hat;del_hat;gam_hat];

    if nargout>1
        cl = -1*fval;

        p_tmp2 = p_tmp;

        if cl == 0
            cl = NaN;
        end

        %% Bootstrap std errs
        s_boot = nan(B,length(p_tmp2));
        for iB = 1:B
            if iB == 1
                [x_b,t_b,bo] = simulate_IVT(beta_hat,dgp_num,N,dt);
            else
                x_b = simulate_IVT(beta_hat,dgp_num,N,dt,bo);
            end

            [tmp1,tmp2] = helpfct_c_pair_Poisson_IG_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
            s_boot(iB,:) = -tmp2';
        end

        %% AVAR calcs
        V = cov(s_boot/sqrt(N));
        H = -1*hessian/n;
        G_inv = (H\V)/H;

        std_err = [exp(p_tmp2(1));exp(p_tmp2(2));exp(p_tmp2(3))].*sqrt(diag(G_inv))/sqrt(n);


        if nargout>3
            aic = cl + trace( V\H );
            bic = cl + 0.5*log(n)*trace( V\H );
        end
    end
        
%% Poisson-Gamma
elseif dgp_num == 3

    if nargin < 5 || isempty(param0)
        param0 = log([mean(y);1;1]);
    end

    %%% Estimate
    if use_grad == 1
        loglik_temp = @(par)( helpfct_c_pair_Poisson_GAM_v10(y,par,Hvec,length(Hvec),n,dt) );
    else
        loglik_temp = @(par)( -c_pair_Poisson_GAM_v10(y,par,Hvec,length(Hvec),n,dt) );
    end
    
    optFct = @(par)( loglik_temp(par) );

    if use_grad == 1
        [p_tmp,fval,~,~,~,hessian] = fminunc(optFct,param0,options);
    else
        p_tmp = fminunc(optFct,param0,options);
    end

    nu_hat  = exp(p_tmp(1));
    H_hat   = exp(p_tmp(2));
    alp_hat = exp(p_tmp(3));
    beta_hat = [nu_hat;H_hat;alp_hat];

    if nargout>1
        cl = -1*fval;

        p_tmp2 = p_tmp;

        if cl == 0
            cl = NaN;
        end

        %% Bootstrap std errs
        s_boot = nan(B,length(p_tmp2));
        for iB = 1:B
            if iB == 1
                [x_b,t_b,bo] = simulate_IVT(beta_hat,dgp_num,N,dt);
            else
                x_b = simulate_IVT(beta_hat,dgp_num,N,dt,bo);
            end

            [tmp1,tmp2] = helpfct_c_pair_Poisson_GAM_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
            s_boot(iB,:) = -tmp2';
        end

        %% AVAR calcs
        bval_hat = max(1,2-H_hat);
        
        V = cov(s_boot/N^(bval_hat/2));
        H = -1*hessian/n;
        G_inv = (H\V)/H;

        std_err = [exp(p_tmp2(1));exp(p_tmp2(2));exp(p_tmp2(3))].*sqrt(diag(G_inv))/n^(1-bval_hat/2);


        if nargout>3
            aic = cl + trace( V\H );
            bic = cl + 0.5*log(n)*trace( V\H );
        end
    end   


%%%%%%%%%%%%%%%%%%%%%%%%%%%% NB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NB-Exponential
elseif dgp_num == 4

    if nargin < 5 || isempty(param0)
        param0 = [log(mean(y));0;0];
    end

    
    %%% Estimate
    if use_grad == 1
        loglik_temp = @(par)( helpfct_c_pair_NB_Exp_v10(y,par,Hvec,length(Hvec),n,dt) );
    else
        loglik_temp = @(par)( -c_pair_NB_Exp_v10(y,par,Hvec,length(Hvec),n,dt) );
    end
    
    optFct = @(par)( loglik_temp(par) );

    if use_grad == 1
        [p_tmp,fval,~,~,~,hessian] = fminunc(optFct,param0,options);
    else
        p_tmp = fminunc(optFct,param0,options);
    end

    m_hat    = exp(p_tmp(1));
    p_hat    = sigmoid(p_tmp(2));
    lam_hat  = exp(p_tmp(3));
    beta_hat = [m_hat;p_hat;lam_hat];

    if nargout>1
        cl = -1*fval;

        p_tmp2 = p_tmp;

        if cl == 0
            cl = NaN;
        end

        s_boot = nan(B,length(p_tmp2));
        for iB = 1:B
            if iB == 1
                [x_b,t_b,bo] = simulate_IVT(beta_hat,dgp_num,N,dt);
            else
                x_b = simulate_IVT(beta_hat,dgp_num,N,dt,bo);
            end

            [tmp1,tmp2] = helpfct_c_pair_NB_Exp_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
            s_boot(iB,:) = -tmp2';
        end

        %% AVAR calcs
        V = cov(s_boot/sqrt(N));
        H = -1*hessian/n;
        G_inv = (H\V)/H;

        std_err = [exp(p_tmp2(1));sigmoid(p_tmp2(2))^2*exp(-p_tmp2(2));exp(p_tmp2(3))].*sqrt(diag(G_inv))/sqrt(n);


        if nargout>3
            aic = cl + trace( V\H );
            bic = cl + 0.5*log(n)*trace( V\H );
        end
    end

%% NB-IG
elseif dgp_num == 5

    if nargin < 5 || isempty(param0)
        param0 = [log(mean(y));0;0;0];
    end
    
    %%% Estimate
    if use_grad == 1
        loglik_temp = @(par)( helpfct_c_pair_NB_IG_v10(y,par,Hvec,length(Hvec),n,dt) );
    else
        loglik_temp = @(par)( -c_pair_NB_IG_v10(y,par,Hvec,length(Hvec),n,dt) );
    end
    
    optFct = @(par)( loglik_temp(par) );

    if use_grad == 1
        [p_tmp,fval,~,~,~,hessian] = fminunc(optFct,param0,options);
    else
        p_tmp = fminunc(optFct,param0,options);
    end

    m_hat    = exp(p_tmp(1));
    p_hat    = sigmoid(p_tmp(2));    
    del_hat  = exp(p_tmp(3));
    gam_hat  = exp(p_tmp(4));
    beta_hat = [m_hat;p_hat;del_hat;gam_hat];

    if nargout>1
        cl = -1*fval;

        p_tmp2 = p_tmp;

        if cl == 0
            cl = NaN;
        end

        %% Bootstrap std errs
        s_boot = nan(B,length(p_tmp2));
        for iB = 1:B
            if iB == 1
                [x_b,t_b,bo] = simulate_IVT(beta_hat,dgp_num,N,dt);
            else
                x_b = simulate_IVT(beta_hat,dgp_num,N,dt,bo);
            end

            [tmp1,tmp2] = helpfct_c_pair_NB_IG_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
            s_boot(iB,:) = -tmp2';
        end

        %% AVAR calcs
        V = cov(s_boot/sqrt(N));
        H = -1*hessian/n;
        G_inv = (H\V)/H;

        std_err = [exp(p_tmp2(1));sigmoid(p_tmp2(2))^2*exp(-p_tmp2(2));exp(p_tmp2(3));exp(p_tmp2(4))].*sqrt(diag(G_inv))/sqrt(n);


        if nargout>3
            aic = cl + trace( V\H );
            bic = cl + 0.5*log(n)*trace( V\H );
        end
    end

%% NB-Gamma
elseif dgp_num == 6

    if nargin < 5 || isempty(param0)
        param0 = [log(mean(y));0;0;0];
    end

    
    %%% Estimate
    if use_grad == 1
        loglik_temp = @(par)( helpfct_c_pair_NB_GAM_v10(y,par,Hvec,length(Hvec),n,dt) );
    else
        loglik_temp = @(par)( -c_pair_NB_GAM_v10(y,par,Hvec,length(Hvec),n,dt) );
    end
    
    optFct = @(par)( loglik_temp(par) );

    if use_grad == 1
        [p_tmp,fval,~,~,~,hessian] = fminunc(optFct,param0,options);
    else
        p_tmp = fminunc(optFct,param0,options);
    end

    m_hat    = exp(p_tmp(1));
    p_hat    = sigmoid(p_tmp(2));    
    H_hat  = exp(p_tmp(3));
    alp_hat  = exp(p_tmp(4));
    beta_hat = [m_hat;p_hat;H_hat;alp_hat];

    if nargout>1
        cl = -1*fval;

        p_tmp2 = p_tmp;

        if cl == 0
            cl = NaN;
        end

        %% Bootstrap std errs
        s_boot = nan(B,length(p_tmp2));
        for iB = 1:B
            if iB == 1
                [x_b,t_b,bo] = simulate_IVT(beta_hat,dgp_num,N,dt);
            else
                x_b = simulate_IVT(beta_hat,dgp_num,N,dt,bo);
            end

            [tmp1,tmp2] = helpfct_c_pair_NB_GAM_v10(x_b,p_tmp2,Hvec,length(Hvec),N,dt);
            s_boot(iB,:) = -tmp2';
        end

        %% AVAR calcs
        
        bval_hat = max(1,2-H_hat);
        
        V = cov(s_boot/N^(bval_hat/2));
        H = -1*hessian/n;
        G_inv = (H\V)/H;

        std_err = [exp(p_tmp2(1));sigmoid(p_tmp2(2))^2*exp(-p_tmp2(2));exp(p_tmp2(3));exp(p_tmp2(4))].*sqrt(diag(G_inv))/n^(1-bval_hat/2);


        if nargout>3
            aic = cl + trace( V\H );
            bic = cl + 0.5*log(n)*trace( V\H );
        end
    end    
end

