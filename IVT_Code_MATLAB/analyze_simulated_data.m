%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script will illustrate the methods of the paper on a simulated data set. 
%
% In particular, this script will illustrate the use of the functions:
% estimate_IVT.m
% estimate_IVT.m (with and without standard errors (SEs) as output).
% modelselect_IVT.m
% forecast_IVT.m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% (c) Mikkel Bennedsen (2021)
%
% This code can be used, distributed, and changed freely. Please cite Bennedsen,
% Lunde, Shephard, and Veraart (2021): "Inference and forecasting for continuous 
% time integer-valued trawl processes and their use in financial economics".
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all;
cd(fileparts(which('analyze_simulated_data.m')));
addpath(genpath('Functions'));
%% Initialization
rng(42);

n = 2000;    % Number of observations to be simulated.
dt = 0.10;   % Equidistant time between observations.
DGP_num = 6; % DGP to be simulated. 1: Poisson-Exp. 2: Poisson-IG. 3: Poisson-Gamma. 4: NB-Exp. 5: NB-IG. 6: NB-Gamma.

K = 10;  % Number of different pairwise observations to include in CL estimator.
N = 100; % Number of observations simulated for every bootstrap replication.
B = 100; % Number of bootstrap replications used in calculation of standard errors.

%% Misc settings
lag = 30;      % For plotting ACF (max lag).
fh = 20;       % For forecasting (max horizon).
xMax = 60;     % For forecasting (max value for prediction).
n_oos = 100;   % Number of observations in out-of-sample forecasting exercise.

%% Parameter settings (here, similar to Table 1 of the main paper)
if DGP_num == 1
    beta0 = [17.50;1.80];
elseif DGP_num == 2
    beta0 = [17.50;1.80;0.80];
elseif DGP_num == 3
    beta0 = [17.50;1.70;0.80];
elseif DGP_num == 4
    beta0 = [7.50;0.70;1.80];
elseif DGP_num == 5
    beta0 = [7.50;0.70;1.80;0.80];
elseif DGP_num == 6
    beta0 = [7.50;0.70;1.70;0.80];
end
%% Build mex files for C++ execution
IVT_build_mex_files;

%% Simulate data
DGP_str = {'Poisson-Exp','Poisson-IG','Poisson-Gamma','NB-Exp','NB-IG','NB-Gamma'};

disp('----------------------------------------------------------------------');
disp(' ');
disp(['Simulating IVT data set using DGP: ',DGP_str{DGP_num},'. Number of observations: ',num2str(n),'.']);

tic
[y,t] = simulate_IVT(beta0,DGP_num,n,dt);
time_estimation0 = toc;

disp(' ');
disp(['Done simulating. Computation time for simulation procedure: ',num2str(time_estimation0),' seconds.']);
disp(' ');
disp('----------------------------------------------------------------------');
%% Estimate all six models
disp(' ');
disp('Estimating six IVT models (no standard errors)...');

tic
%% Estimate Poisson-Exp model
beta_hat_PExp   = estimate_IVT(y,dt,K,1);
disp('1. Poisson-Exp done...');
%% Estimate Poisson-IG model
beta_hat_PIG     = estimate_IVT(y,dt,K,2);
disp('2. Poisson-IG done...');
%% Estimate Poisson-GAM model
beta_hat_PGAM   = estimate_IVT(y,dt,K,3);
disp('3. Poisson-Gamma done...');
%% Estimate NB-Exp model
beta_hat_NBExp = estimate_IVT(y,dt,K,4);
disp('4. NB-Exp done...');
%% Estimate NB-IG model
beta_hat_NBIG   = estimate_IVT(y,dt,K,5);
disp('5. NB-IG done...');
%% Estimate NB-GAM model
beta_hat_NBGAM = estimate_IVT(y,dt,K,6);
disp('6. NB-Gamma done...');

time_estimation1 = toc;

disp(' ');
disp(['Done estimating. Computation time for estimation procedure (no standard errors): ',num2str(time_estimation1),' seconds.']);
disp(' ');
disp('----------------------------------------------------------------------');
%% Estimate with standard errors (much slower)
disp(' ');
disp('Estimating six IVT models (with standard errors)...');

tic
%% Estimate Poisson-Exp model
[beta_hat_PExp,se_PExp]   = estimate_IVT(y,dt,K,1,[],[],B,N);
disp('1. Poisson-Exp done...');
%% Estimate Poisson-IG model
[beta_hat_PIG,se_PIG]     = estimate_IVT(y,dt,K,2,[],[],B,N);
disp('2. Poisson-IG done...');
%% Estimate Poisson-GAM model
[beta_hat_PGAM,se_PGAM]   = estimate_IVT(y,dt,K,3,[],[],B,N);
disp('3. Poisson-Gamma done...');
%% Estimate NB-Exp model
[beta_hat_NBExp,se_NBExp] = estimate_IVT(y,dt,K,4,[],[],B,N);
disp('4. NB-Exp done...');
%% Estimate NB-IG model
[beta_hat_NBIG,se_NBIG]   = estimate_IVT(y,dt,K,5,[],[],B,N);
disp('5. NB-IG done...');
%% Estimate NB-GAM model
[beta_hat_NBGAM,se_NBGAM] = estimate_IVT(y,dt,K,6,[],[],B,N);
disp('6. NB-Gamma done...');

time_estimation2 = toc;

disp(' ');
disp(['Done estimating. Computation time for estimation procedure (with standard errors): ',num2str(time_estimation2),' seconds.']);
disp(' ');
disp('----------------------------------------------------------------------');
%% Model selection
disp(' ');
disp('Performing model selection...');

tic
[model_num,IC] = modelselect_IVT(y,dt,K,B,N);
time_modelselection = toc;

disp(' ');
disp(['Done selecting. Computation time for model selection procedure: ',num2str(time_modelselection),' seconds.']);
disp(' ');
disp('----------------------------------------------------------------------');
%% Forecasting
n_is = n-n_oos; % Last in-sample period.
disp(' ');
disp(['Forecasting ',num2str(fh),' periods ahead using Poisson-Exp and ',DGP_str{DGP_num},'. In-sample data are y(1:',num2str(n_is),') and y(',num2str(n_is),') = ',num2str(y(n_is)),'...']);
tic
condForecastProb_IVT_PExp  = forecast_IVT(y(1:n_is),fh,dt,1,xMax,[],K);
condForecastProb_IVT_DGP = forecast_IVT(y(1:n_is),fh,dt,DGP_num,xMax,[],K);
time_forecast = toc;
disp(' ');
disp(['Done forecasting. Computation time for forecasting procedure: ',num2str(time_forecast),' seconds.']);
disp(' ');
disp('----------------------------------------------------------------------');

%% Construct table with parameters estimates+SEs
estRes = nan(12,8);

estRes(1,1) = beta_hat_PExp(1); estRes(2,1) = se_PExp(1);
estRes(1,4) = beta_hat_PExp(2); estRes(2,4) = se_PExp(2);

estRes(3,1) = beta_hat_PIG(1); estRes(4,1) = se_PIG(1);
estRes(3,5) = beta_hat_PIG(2); estRes(4,5) = se_PIG(2);
estRes(3,6) = beta_hat_PIG(3); estRes(4,6) = se_PIG(3);

estRes(5,1) = beta_hat_PGAM(1); estRes(6,1) = se_PGAM(1);
estRes(5,7) = beta_hat_PGAM(2); estRes(6,7) = se_PGAM(2);
estRes(5,8) = beta_hat_PGAM(3); estRes(6,8) = se_PGAM(3);

estRes(7,2) = beta_hat_NBExp(1); estRes(8,2) = se_NBExp(1);
estRes(7,3) = beta_hat_NBExp(2); estRes(8,3) = se_NBExp(2);
estRes(7,4) = beta_hat_NBExp(3); estRes(8,4) = se_NBExp(3);

estRes(9,2) = beta_hat_NBIG(1); estRes(10,2) = se_NBIG(1);
estRes(9,3) = beta_hat_NBIG(2); estRes(10,3) = se_NBIG(2);
estRes(9,5) = beta_hat_NBIG(3); estRes(10,5) = se_NBIG(3);
estRes(9,6) = beta_hat_NBIG(4); estRes(10,6) = se_NBIG(4);

estRes(11,2) = beta_hat_NBGAM(1); estRes(12,2) = se_NBGAM(1);
estRes(11,3) = beta_hat_NBGAM(2); estRes(12,3) = se_NBGAM(2);
estRes(11,7) = beta_hat_NBGAM(3); estRes(12,7) = se_NBGAM(3);
estRes(11,8) = beta_hat_NBGAM(4); estRes(12,8) = se_NBGAM(4);
%% Print to screen
disp(' ');
disp('Information criteria:');
disp([IC.CL;IC.CLAIC;IC.CLBIC]);
disp(' ');
[val,indx] = max([IC.CL;IC.CLAIC;IC.CLBIC]');
disp('Selected model:      CL             CLAIC          CLBIC');
disp(['                     ',DGP_str{indx(1)},'     ',DGP_str{indx(2)},'     ',DGP_str{indx(3)}]);
disp(' ');
disp('Estimation results:');
disp(estRes);

%% plot data
stp = max(y);

fig1 = figure;
subplot(7,2,[1,2]);
stairs(t,y+1), hold on 
plot(t(n_is)*ones(100,1),linspace(0,max(y+1)+1,100),'r-'), hold on
axis([t(1),t(end),0,max(y)+2]);
title(['Simulated ',DGP_str{DGP_num},' IVT process'],'Interpreter','latex','FontSize',10);
grid on

for i = 1:6
    if i == 1
        nu_hat = beta_hat_PExp(1);
        lam_hat = beta_hat_PExp(2);
        
        EstAcf = exp(-lam_hat*((1:lag)'*dt));
        EstNBinProbs = poisspdf(0:stp,nu_hat/lam_hat);
        
        mod_str = {'Poisson-Exp'};
    elseif i == 2
        nu_hat = beta_hat_PIG(1);
        del_hat = beta_hat_PIG(2);
        gam_hat = beta_hat_PIG(3);
        
        EstAcf = exp(del_hat*gam_hat*(1-sqrt(1+2*((1:lag)'*dt)/gam_hat^2)));
        EstNBinProbs = poisspdf(0:stp,nu_hat*gam_hat/del_hat);
        
        mod_str = {'Poisson-IG'};
    elseif i == 3
        nu_hat = beta_hat_PGAM(1);
        H_hat = beta_hat_PGAM(2);
        alp_hat = beta_hat_PGAM(3);
        
        EstAcf = (1 + ((1:lag)'*dt)/alp_hat ).^(-H_hat);
        EstNBinProbs = poisspdf(0:stp,nu_hat*(alp_hat/H_hat));
        
        mod_str = {'Poisson-Gamma'};
    elseif i == 4
        m_hat = beta_hat_NBExp(1);
        p_hat = beta_hat_NBExp(2);
        lam_hat = beta_hat_NBExp(3);
        
        EstAcf = exp(-lam_hat*((1:lag)'*dt));
        EstNBinProbs = nbinpdf(0:stp,m_hat/lam_hat,1-p_hat);
        
        mod_str = {'NB-Exp'};
    elseif i == 5
        m_hat = beta_hat_NBIG(1);
        p_hat = beta_hat_NBIG(2);
        del_hat = beta_hat_NBIG(3);
        gam_hat = beta_hat_NBIG(4);
        
        EstAcf = exp(del_hat*gam_hat*(1-sqrt(1+2*((1:lag)'*dt)/gam_hat^2)));
        EstNBinProbs = nbinpdf(0:stp,m_hat/(del_hat/gam_hat),1-p_hat);
        
         mod_str = {'NB-IG'};
    elseif i == 6
        m_hat = beta_hat_NBGAM(1);
        p_hat = beta_hat_NBGAM(2);
        H_hat = beta_hat_NBGAM(3);
        alp_hat = beta_hat_NBGAM(4);
        
        EstAcf = (1 + ((1:lag)'*dt)/alp_hat ).^(-H_hat);
        EstNBinProbs = nbinpdf(0:stp,m_hat*(alp_hat/H_hat),1-p_hat);  
        
         mod_str = {'NB-Gamma'};
    else
        
        asdf
        
    end
    
    subplot(7,2,i*2+1);
    [xout,nn] = hist(y,0:stp);
    h = bar(nn+1,xout/size(y,1)); hold on
    h = plot([0:stp]+1,EstNBinProbs,'ro-', 'LineWidth',1.5,'MarkerSize',4);%, 'MarkerFaceColor',[.49 1 .63]); hold off;

    xlim([0 stp+2]);
    ylim([0,0.2]);
    
    if i == 1
        title('Marginal distribution','Interpreter','latex','FontSize',10);
    end
    ylabel(mod_str,'Interpreter','latex','FontSize',10);
    if i == 6
       xlabel('Spread level','Interpreter','latex','FontSize',10);
    end
    %% Plot autocorrelation
    subplot(7,2,(i+1)*2);
    sACF = sacf(y,lag,0,0); 
    h = bar(1:lag,sACF); hold on
    h = plot(1:lag,EstAcf,'-r', 'LineWidth',2,'MarkerSize',19); hold off;
    ylim([-0.2 1]);
    xlim([0 lag]);
    %xlabel('ACF');
    set(gca,'XTick',[1 5:5:30]);
    set(gca,'XTickLabel',[1 5:5:30]);
    if i == 1
        title('Autocorrelation','Interpreter','latex','FontSize',10);
    end
    if i == 6
       xlabel('Lag','Interpreter','latex','FontSize',10);
    end


end

%% Plot predictive distributions
hvec = [1,2,5,10];
figure;
for i = 1:4
    subplot(2,2,i);
    bar(0:xMax,condForecastProb_IVT_PExp(hvec(i),:));
    title(['Poisson-Exp ($h = ',num2str(hvec(i)),'$)'],'Interpreter','latex','FontSize',10);
    xlabel('$y(T+h)$','Interpreter','latex','FontSize',10);
    ylabel(['Predictive probability conditional on $y(T) = ',num2str(y(n_is)),'$'],'Interpreter','latex','FontSize',10);
end

figure;
for i = 1:4
    subplot(2,2,i);
    bar(0:xMax,condForecastProb_IVT_DGP(hvec(i),:));
    title([DGP_str{DGP_num},' ($h = ',num2str(hvec(i)),'$)'],'Interpreter','latex','FontSize',10);
    xlabel('$y(T+h)$','Interpreter','latex','FontSize',10);
    ylabel(['Predictive probability conditional on $y(T) = ',num2str(y(n_is)),'$'],'Interpreter','latex','FontSize',10);
end
