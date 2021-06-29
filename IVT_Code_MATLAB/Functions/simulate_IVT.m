function [x,time2,burnIn] = simulate_IVT(params,dgp_num,n,dt,burnIn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulate IVT process on equidistant grid.
%
% INPUT
% params     : Parameters. Depends on specification of DGP (see below)
% dgp_num:   : IVT DGP. 1: Poisson-Exp. 2: Poisson-IG. 3: Poisson-Gamma. 4: NB-Exp. 5: NB-IG. 6: NB-Gamma.
% n          : Number of observations.
% dt         : Time between (equidistant) observations.
% burnOut    : Initial burn-in period. (Default set so that thereoretical ACF(burnOut) < ACF_cutoff, where default is ACF_cutoff=0.01.)
%
% OUTPUT
% x          : (n x 1) vector with observations of IVT process
% time2      : (n x 1) vector time vector (optional)
% burnOut    : Burnout period (optional). Can be output and input again for
%              speed in large Monte Carlo studies.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% (c) Mikkel Bennedsen (2021)
%
% This code can be used, distributed, and changed freely. Please cite Bennedsen,
% Lunde, Shephard, and Veraart (2021): "Likelihood-based estimation and
% forecasting fpr continuous time integer-valued trawl processes".
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Specify DGP
if dgp_num == 1
    useSeed = 'Poisson';
    useTrawl = 'Exp';
elseif dgp_num == 2
    useSeed = 'Poisson';
    useTrawl = 'IG';
elseif dgp_num == 3
    useSeed = 'Poisson';
    useTrawl = 'Gamma';    
elseif dgp_num == 4
    useSeed = 'NB';
    useTrawl = 'Exp';
elseif dgp_num == 5
    useSeed = 'NB';
    useTrawl = 'IG';
elseif dgp_num == 6
    useSeed = 'NB';
    useTrawl = 'Gamma';
else
    error('Model not implemented');
end

%%% Further inft on Levy seed and trawl function specifications %%%
% useSeed:  Distribution of Lévy seed, determining the marginal
%           distribution of the trawl
%   'Poisson':  Poisson Lévy seed.          Params: (int)            [intensity]
%   'Skellam':  Skellam Lévy seed.          Params: (int,p)          [intensity, Prob(up move)]
%   'NB'     :  Negative bin. Lévy seed.    Params: (int,m,p)        [intensity,# fail, p succes]
%   'DNB'    :  Delta Neg. Bin. Lévy seed.  Params: (int,m,p)        [intensity,# fail, p succes]
%
% useTrawl:  Trawl function.
%   'Exp'    :  Exponential trawl.          Params: (lambda)
%   'supExp' :  Finite sup exp. trawls.     Params: (w1,lam1, w2,lam2, ...)
%   'GIG'    :  Sup-GIG trawl.              Params: (nu,gam,del)
%   'IG'     :  Sup-IG trawl.               Params: (gam,del)       [Note: GIG trawl with vu = 1/2]
%   'Gamma'  :  Gamma trawl.                Params: (H,alph)        [H > 0. H < 1 => Long memory]

%% Make coloumn vector
if size(params,2) > 1
    params = params'; 
end

ACF_cutoff = 0.01; % Used when deciding burn-in period.

T = n*dt;
%% Load in Levy seed parameters

pC = 1;             % Counter keeping track of number of params.
switch useSeed
    case 'Poisson'
        intens = params(pC); pC = pC+1; 
    case 'Skellam'
        intens = params(pC); pC = pC+1;
        pUp = params(pC); pC = pC+1;
    case 'NB'
        mP = params(pC); pC = pC+1;
        pP = params(pC); pC = pC+1;
        
        intens = mP*abs(log(1-pP));
    case 'DNB'
        mP = params(pC); pC = pC+1;
        pP = params(pC); pC = pC+1;
        error('Not implemented yet');
    otherwise
        error('Wrong Lévy seed specification');
end    

%% Load in trawl function
switch useTrawl
    case 'Exp'
        lambda = params(pC); pC = pC+1;
        gFct = @(x) ( exp(-lambda*x) );
    case 'supExp'
        W = params(pC:2:end-1);
        L = params((pC+1):2:end);
        gFct = @(x) ( sum(W.*exp(-L.*x'))' ); %Note: params should be Px1.
    case 'GIG'
        nu = params(pC);  pC = pC+1;        
        del = params(pC); pC = pC+1;
        gam = params(pC); pC = pC+1;
        gFct = @(x) ( (1+2*x/(gam^2)).^(-0.5*nu) .* besselk(nu,del*gam*sqrt(1+2*x/(gam^2)))/besselk(nu,del*gam) );
    case 'IG'       
        del = params(pC); pC = pC+1;
        gam = params(pC); pC = pC+1;
        gFct = @(x) ( exp(del*gam*(1-sqrt(1+2*x/(gam^2))))./sqrt(1 + 2*x/(gam^2)) );
    case 'Gamma'
        H = params(pC);    pC = pC+1;
        alph = params(pC); pC = pC+1;
        gFct = @(x) ( (1+x/alph).^(-(H+1)) );
    otherwise
        error('Wrong trawl function');
end

%% Check burnout period
if nargin < 5
    switch useTrawl
        case 'Exp'
            theoretical_acf = @(x) ( exp(-lambda*x) );
        case 'supExp'
            theoretical_acf = @(x) ( sum(W.*exp(-L.*x))' ); %Note: params should be Px1.
        case 'GIG'
            theoretical_acf = @(x) ( sqrt(1+2*x/(gam^2)).^(1-nu).*besselk(nu-1,del*gam*sqrt(1+2*x/(gam^2)))./besselk(nu-1,del*gam) );
        case 'IG'       
            theoretical_acf = @(x) ( exp(del*gam*(1-sqrt(1+2*x/(gam^2)))) );
        case 'Gamma'
            theoretical_acf = @(x) ( (1+x/alph).^(-H) );
        otherwise
            error('Wrong trawl function');
    end    
    
    [~,tmp_indx] = find(theoretical_acf(dt:dt:T) < ACF_cutoff );
    if isempty(tmp_indx)
        warning(['Simulated ',useSeed,'-',useTrawl,' IVT process is very persistent. Number of burn-in observations set to burnIn=2n. Stationarity might not have been reached at end of burn-in period.']);
        burnIn = 2*T;
    else
        burnIn = dt*tmp_indx(1);
    end
end


%% Initialization
TT = burnIn + T;  % Total time to simulate.
N = floor(TT/dt);
time = dt:dt:N*dt;

%% Simulate marks
switch useSeed
    case 'Poisson'
        R = poissrnd(TT*intens);    % # Marks
        CC = ones(R,1);
    case 'Skellam'
        R = poissrnd(TT*intens);    % # Marks
        CC = 1-2*(rand(R,1) > pUp );
    case 'NB'
        R = poissrnd(TT*intens);    % # Marks
        CC = ranLogadll2(pP,R,1);
    case 'DNB'
        error('Not implemented yet');
    otherwise
        error('Wrong Lévy seed specification');
end   

%% Simulate Arrival times and Marks
tau = TT*rand(R,1); % Arrival times.
U = rand(R,1);      % Height of marks.

%% Generate path
X = sum( repmat(CC,1,N).*(tau<time).*(gFct(time-repmat(tau,1,N))>repmat(U,1,N)) )';

%% Discard burn-in period for output
x = X(end-n+1:end);
time2 = (dt:dt:n*dt)';