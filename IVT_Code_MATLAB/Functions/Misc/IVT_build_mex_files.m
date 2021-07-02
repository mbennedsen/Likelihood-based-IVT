%%% Function to built mex-files

%% Poisson-Exp
if ~(exist('c_pair_Poisson_Exp_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_Poisson_Exp_v10.c')));
    mex c_pair_Poisson_Exp_v10.c;
    cd(dir0);
end

if ~(exist('c_pair_grad_Poisson_Exp_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_grad_Poisson_Exp_v10.c')));
    mex c_pair_grad_Poisson_Exp_v10.c;
    cd(dir0);
end


%% Poisson-IG
if ~(exist('c_pair_Poisson_IG_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_Poisson_IG_v10.c')));
    mex c_pair_Poisson_IG_v10.c;
    cd(dir0);
end

if ~(exist('c_pair_grad_Poisson_IG_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_grad_Poisson_IG_v10.c')));
    mex c_pair_grad_Poisson_IG_v10.c;
    cd(dir0);
end

%% Poisson-GAM
if ~(exist('c_pair_Poisson_GAM_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_Poisson_GAM_v10.c')));
    mex c_pair_Poisson_GAM_v10.c;
    cd(dir0);
end


if ~(exist('c_pair_grad_Poisson_GAM_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_grad_Poisson_GAM_v10.c')));
    mex c_pair_grad_Poisson_GAM_v10.c;
    cd(dir0);
end


%% NB-Exp
if ~(exist('c_pair_NB_Exp_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_NB_Exp_v10.c')));
    mex c_pair_NB_Exp_v10.c;
    cd(dir0);
end

if ~(exist('c_pair_grad_NB_Exp_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_grad_NB_Exp_v10.c')));
    mex c_pair_grad_NB_Exp_v10.c;
    cd(dir0);
end

%% NB-IG
if ~(exist('c_pair_NB_IG_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_NB_IG_v10.c')));
    mex c_pair_NB_IG_v10.c;
    cd(dir0);
end

if ~(exist('c_pair_grad_NB_IG_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_grad_NB_IG_v10.c')));
    mex c_pair_grad_NB_IG_v10.c;
    cd(dir0);
end

%% NG-GAM
if ~(exist('c_pair_NB_GAM_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_NB_GAM_v10.c')));
    mex c_pair_NB_GAM_v10.c;
    cd(dir0);
end

if ~(exist('c_pair_grad_NB_GAM_v10.mexmaci64')==3)
    dir0 = pwd;
    cd(fileparts(which('c_pair_grad_NB_GAM_v10.c')));
    mex c_pair_grad_NB_GAM_v10.c;
    cd(dir0);
end
