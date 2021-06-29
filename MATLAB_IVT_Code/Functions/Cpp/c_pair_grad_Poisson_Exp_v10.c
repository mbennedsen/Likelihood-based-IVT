#include <math.h>
#include <limits.h>
#include "mex.h"
#include "matrix.h"

double fac(double a) 
{ 
    double answer = 1.0; 
    while(a>1) 
    answer *= a--; 
    return answer; 
} 

double pPoi(double y, double lam) 
{ 
    return pow(lam,y)*exp(-lam)/fac(y);
}

double Acom(double t, double lam)
{
    return  exp(-lam*t)/lam;
}

double Aunc(double t, double lam)
{
    return  (1-exp(-lam*t))/lam;
}

double Acom_d_lambda(double t, double lam)
{
    return -exp(-lam*t)*(1/lam + t)/lam;
}

double Aunc_d_lambda(double t, double lam)
{
    return  (t*exp(-lam*t) - (1-exp(-lam*t))/lam)/lam;
}

void c_pair_grad_Poisson_Exp_v10(double *Y, double *par, double *hVec, int h, int N, double dt, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, Leb_com, Leb_unc, dlam_Leb_com, dlam_Leb_unc, temp, temp_dnu, temp_dlam, lambda, nu, Yt, Ys, P1, P2, P3;
    
    nu    = exp(par[0]);
    lambda = exp(par[1]);
    
    fts[0] = 0; // loglik
    fts[1] = 0; // grad-nu
    fts[2] = 0; // grad-lam
    for (ih=0; ih<h; ih++){
        hh = (int)hVec[ih];
        
        Leb_com = Acom(hh*dt,lambda);
        Leb_unc = Aunc(hh*dt,lambda);
        
        dlam_Leb_com = Acom_d_lambda(hh*dt,lambda);
        dlam_Leb_unc = Aunc_d_lambda(hh*dt,lambda);
        
        A1 = nu*Leb_com;
        A2 = nu*Leb_unc;
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            temp_dnu  = 0;
            temp_dlam = 0;
            for (j=0; j<c+1; j++){
                P1 = pPoi(Yt-j,A2);
                P2 = pPoi(Ys-j,A2);
                P3 = pPoi(j,A1);
                
                temp = temp + P1*P2*P3;
                
                temp_dnu  = temp_dnu  + P1*((Yt-j)/nu - Leb_unc)*P2*P3 + P1*P2*((Ys-j)/nu - Leb_unc)*P3 + P1*P2*P3*(j/nu - Leb_com);
                temp_dlam = temp_dlam + P1*dlam_Leb_unc*((Yt-j)/Leb_unc - nu)*P2*P3 + P1*P2*dlam_Leb_unc*((Ys-j)/Leb_unc - nu)*P3 + P1*P2*P3*dlam_Leb_com*(j/Leb_com - nu);
            }
            if (temp>0){
                fts[0] = fts[0] + log(temp); 
                fts[1] = fts[1] + temp_dnu/temp; 
                fts[2] = fts[2] + temp_dlam/temp; 
            }
        }
    }
}

 
/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *par, *hVec, *getH, *getN, *getDT, *Y, *fts;
	int N, h;
    double dt;

	/*  Check for proper number of arguments. */
	if(nrhs!=6) mexErrMsgTxt("Six inputs required.");
	if(nlhs!=1) mexErrMsgTxt("One output required."); 

	/*  Create a pointer to the input matrices . */

    Y           = mxGetPr(prhs[0]);
    par         = mxGetPr(prhs[1]);
    hVec        = mxGetPr(prhs[2]);
    getH        = mxGetPr(prhs[3]);
    h           = (int)getH[0];
	getN        = mxGetPr(prhs[4]);
    N           = (int)getN[0];
	getDT       = mxGetPr(prhs[5]);
    dt          = (double)getDT[0];
            
	/*  Set the output pointer to the output matrix. */
   	plhs[0] = mxCreateDoubleMatrix((int)3,1, mxREAL);

	/*  Create a C pointer to a copy of the output matrix. */
	fts = mxGetPr(plhs[0]);
	
    //	printf("Number of rows is %d, number of cols is %d\n",T,T);

    c_pair_grad_Poisson_Exp_v10(Y,par,hVec,h,N,dt,fts);
}

