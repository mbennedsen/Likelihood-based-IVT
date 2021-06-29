function [predictiveDistributionIVT, forecast_mean] = forecast_IVT(y,fh,dt,dgp_num,yMax,params,K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate IVT process observed on equidistant grid.
%
% INPUT
% y          : Data (equidistant).
% fh         : Maximum forecast horizon (i.e. output will give predictive distribution for horizons h = 1, 2, ..., fh.
% dt:        : Time between (equidistant) observations.
% dgp_num:   : IVT DGP. 1: Poisson-Exp. 2: Poisson-IG. 3: Poisson-Gamma. 4: NB-Exp. 5: NB-IG. 6: NB-Gamma.
% yMax       : Maximum y-value for calculation of predictive distribution (i.e. output will be predictive distribution for the y-values 0, 1, ..., yMax.
% param      : Values for parameters of the DGP. If empty, the parameters will be estimated using the estimate_IVT function.
% K          : Number of different pairwise observations to include in CL estimator. (Default: K = 5.)
%
% OUTPUT
% predictiveDistributionIVT : (fh x (yMax+1)) matrix of predictive probabilities.
% forecast_mean             : (fh x 1) vector of mean forecasts, calculated using predictiveDistributionIVT.
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

if nargin<5
    yMax = max(y) + floor(std(y))+1;
end
    
if nargin < 6 % Then estimate parameters
    K = 5;
    
    params = estimate_IVT(y,dt,K,dgp_num);
end

if isempty(params) % If params is empty, then estimate using Hvec and options
    if nargin < 7
        K = 5;
    end
    params = estimate_IVT(y,dt,K,dgp_num);   
end

%% Poisson-Exp
if dgp_num == 1

    nu_hat  = params(1);
    lam_hat = params(2);
    
    Leb_A0        = @(l)(1/l);
    Leb_intersect = @(t,l)( exp(-l*t)/l );
    Leb_minus     = @(t,l)( (1-exp(-l*t))/l );

    Leb0 = Leb_A0(lam_hat);

    y_n = y(end);

    binomial_vector = nan(1,y_n+1);
    for i = 0:y_n
        binomial_vector(i+1) = nchoosek(y_n,i);
    end

    forecast_mean = nan(fh,1);
    predictiveDistributionIVT = nan(fh,yMax+1);
    for i = 1:fh
        %%% Calculating the conditional probabilities
        Leb_i = Leb_intersect(dt*i,lam_hat);
        Leb_m = Leb_minus(dt*i,lam_hat);
        for j = 0:yMax
            xTmp = min(j,y_n);
            cVec = 0:xTmp;

            %%% CONDITIONAL PROB: IVT
             predictiveDistributionIVT(i,j+1) = sum( (nu_hat*Leb_m).^(j-cVec)./factorial(j-cVec).*exp(-nu_hat*Leb_m).*binomial_vector(1:(xTmp+1)).*(Leb_i/Leb0).^cVec.*(1-Leb_i/Leb0).^(y_n-cVec) );
        end
        forecast_mean(i) = sum( predictiveDistributionIVT(i,:).*(0:yMax) );
    end
    

%% Poisson-IG
elseif dgp_num == 2

    nu_hat  = params(1);
    del_hat = params(2);
    gam_hat = params(3);

    Leb_A0        = @(d,g)(g/d);
    Leb_intersect = @(t,d,g)( g/d*exp(d*g*(1-sqrt(1+2*t/g^2))) );
    Leb_minus     = @(t,d,g)( g/d*(1-exp(d*g*(1-sqrt(1+2*t/g^2)))) );

    Leb0 = Leb_A0(del_hat,gam_hat);

    y_n = y(end);

    binomial_vector = nan(1,y_n+1);
    for i = 0:y_n
        binomial_vector(i+1) = nchoosek(y_n,i);
    end

    forecast_mean = nan(fh,1);
    predictiveDistributionIVT = nan(fh,yMax+1);
    for i = 1:fh
        %%% Calculating the conditional probabilities
        Leb_i = Leb_intersect(dt*i,del_hat,gam_hat);
        Leb_m = Leb_minus(dt*i,del_hat,gam_hat);
        for j = 0:yMax
            xTmp = min(j,y_n);
            cVec = 0:xTmp;

            %%% CONDITIONAL PROB: IVT
             predictiveDistributionIVT(i,j+1) = sum( (nu_hat*Leb_m).^(j-cVec)./factorial(j-cVec).*exp(-nu_hat*Leb_m).*binomial_vector(1:(xTmp+1)).*(Leb_i/Leb0).^cVec.*(1-Leb_i/Leb0).^(y_n-cVec) );
        end
        forecast_mean(i) = sum( predictiveDistributionIVT(i,:).*(0:yMax) );
    end
        
%% Poisson-Gamma
elseif dgp_num == 3

    nu_hat  = params(1);
    H_hat   = params(2);
    alp_hat = params(3);
    
    Leb_A0        = @(h,a)(a/h);
    Leb_intersect = @(t,h,a)( a*(1+t/a).^(-h)/h );
    Leb_minus     = @(t,h,a)( a*(1-(1+t/a).^(-h))/h );

    Leb0 = Leb_A0(H_hat,alp_hat);

    y_n = y(end);

    binomial_vector = nan(1,y_n+1);
    for i = 0:y_n
        binomial_vector(i+1) = nchoosek(y_n,i);
    end

    forecast_mean = nan(fh,1);
    predictiveDistributionIVT = nan(fh,yMax+1);
    for i = 1:fh
        %%% Calculating the conditional probabilities
        Leb_i = Leb_intersect(dt*i,H_hat,alp_hat);
        Leb_m = Leb_minus(dt*i,H_hat,alp_hat);
        for j = 0:yMax
            xTmp = min(j,y_n);
            cVec = 0:xTmp;

            %%% CONDITIONAL PROB: IVT
             predictiveDistributionIVT(i,j+1) = sum( (nu_hat*Leb_m).^(j-cVec)./factorial(j-cVec).*exp(-nu_hat*Leb_m).*binomial_vector(1:(xTmp+1)).*(Leb_i/Leb0).^cVec.*(1-Leb_i/Leb0).^(y_n-cVec) );
        end
        forecast_mean(i) = sum( predictiveDistributionIVT(i,:).*(0:yMax) );
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%% NB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% NB-Exponential
elseif dgp_num == 4

    m_hat    = params(1);
    p_hat    = params(2);
    lam_hat  = params(3);
    
    Leb_A0        = @(l)(1/l);
    Leb_intersect = @(t,l)( exp(-l*t)/l );
    Leb_minus     = @(t,l)( (1-exp(-l*t))/l );

    Leb0 = Leb_A0(lam_hat);

    y_n = y(end);

    binomial_vector = nan(1,y_n+1);
    for i = 0:y_n
        binomial_vector(i+1) = nchoosek(y_n,i);
    end

    forecast_mean = nan(fh,1);
    predictiveDistributionIVT = nan(fh,yMax+1);
    for i = 1:fh
        %%% Calculating the conditional probabilities
        Leb_i = Leb_intersect(dt*i,lam_hat);
        Leb_m = Leb_minus(dt*i,lam_hat);
        for j = 0:yMax
            xTmp = min(j,y_n);
            cVec = 0:xTmp;

            %%% CONDITIONAL PROB: IVT
             predictiveDistributionIVT(i,j+1) = sum( (1-p_hat)^(Leb_m*m_hat).*p_hat.^(j-cVec).*binomial_vector(1:(xTmp+1))./factorial(j-cVec).*( gamma(Leb_m*m_hat + j-cVec)/gamma(Leb_m*m_hat) ).*( gamma(Leb_m*m_hat + y_n-cVec)/gamma(Leb_m*m_hat) ).*( gamma(Leb_i*m_hat+cVec)/gamma(Leb_i*m_hat) ).* ( gamma(Leb0*m_hat)/gamma(Leb0*m_hat + y_n) ) );
        end
        forecast_mean(i) = sum( predictiveDistributionIVT(i,:).*(0:yMax) );
    end
    
%% NB-IG
elseif dgp_num == 5

    m_hat    = params(1);
    p_hat    = params(2);
    del_hat  = params(3);
    gam_hat  = params(4);
    
    Leb_A0        = @(d,g)(g/d);
    Leb_intersect = @(t,d,g)( g/d*exp(d*g*(1-sqrt(1+2*t/g^2))) );
    Leb_minus     = @(t,d,g)( g/d*(1-exp(d*g*(1-sqrt(1+2*t/g^2)))) );

    Leb0 = Leb_A0(del_hat,gam_hat);

    y_n = y(end);

    binomial_vector = nan(1,y_n+1);
    for i = 0:y_n
        binomial_vector(i+1) = nchoosek(y_n,i);
    end

    forecast_mean = nan(fh,1);
    predictiveDistributionIVT = nan(fh,yMax+1);
    for i = 1:fh
        %%% Calculating the conditional probabilities
        Leb_i = Leb_intersect(dt*i,del_hat,gam_hat);
        Leb_m = Leb_minus(dt*i,del_hat,gam_hat);
        for j = 0:yMax
            xTmp = min(j,y_n);
            cVec = 0:xTmp;

            %%% CONDITIONAL PROB: IVT
            predictiveDistributionIVT(i,j+1) = sum( (1-p_hat)^(Leb_m*m_hat).*p_hat.^(j-cVec).*binomial_vector(1:(xTmp+1))./factorial(j-cVec).*( gamma(Leb_m*m_hat + j-cVec)/gamma(Leb_m*m_hat) ).*( gamma(Leb_m*m_hat + y_n-cVec)/gamma(Leb_m*m_hat) ).*( gamma(Leb_i*m_hat+cVec)/gamma(Leb_i*m_hat) ).* ( gamma(Leb0*m_hat)/gamma(Leb0*m_hat + y_n) ) );
        end
        forecast_mean(i) = sum( predictiveDistributionIVT(i,:).*(0:yMax) );
    end
    
%% NB-Gamma
elseif dgp_num == 6
    m_hat    = params(1);
    p_hat    = params(2);
    H_hat    = params(3);
    alp_hat  = params(4);
    
    Leb_A0        = @(h,a)(a/h);
    Leb_intersect = @(t,h,a)( a*(1+t/a).^(-h)/h );
    Leb_minus     = @(t,h,a)( a*(1-(1+t/a).^(-h))/h );

    Leb0 = Leb_A0(H_hat,alp_hat);

    y_n = y(end);

    binomial_vector = nan(1,y_n+1);
    for i = 0:y_n
        binomial_vector(i+1) = nchoosek(y_n,i);
    end

    forecast_mean = nan(fh,1);
    predictiveDistributionIVT = nan(fh,yMax+1);
    for i = 1:fh
        %%% Calculating the conditional probabilities
        Leb_i = Leb_intersect(dt*i,H_hat,alp_hat);
        Leb_m = Leb_minus(dt*i,H_hat,alp_hat);
        for j = 0:yMax
            xTmp = min(j,y_n);
            cVec = 0:xTmp;

            %%% CONDITIONAL PROB: IVT
            predictiveDistributionIVT(i,j+1) = sum( (1-p_hat)^(Leb_m*m_hat).*p_hat.^(j-cVec).*binomial_vector(1:(xTmp+1))./factorial(j-cVec).*( gamma(Leb_m*m_hat + j-cVec)/gamma(Leb_m*m_hat) ).*( gamma(Leb_m*m_hat + y_n-cVec)/gamma(Leb_m*m_hat) ).*( gamma(Leb_i*m_hat+cVec)/gamma(Leb_i*m_hat) ).* ( gamma(Leb0*m_hat)/gamma(Leb0*m_hat + y_n) ) );
        end
        forecast_mean(i) = sum( predictiveDistributionIVT(i,:).*(0:yMax) );
    end
       
end
