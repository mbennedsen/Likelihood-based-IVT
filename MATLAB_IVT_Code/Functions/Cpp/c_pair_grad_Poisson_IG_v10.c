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

double Acom(double t, double d, double g)
{
    return (g/d)*exp(d*g*(1-pow((1+2*t/pow(g,2)),0.5) ));
}

double Aunc(double t, double d, double g)
{
    return (g/d)*(1-exp(d*g*(1-pow((1+2*t/pow(g,2)),0.5) )));
}

double Acom_d_delta(double t, double d, double g)
{
    return  Acom(t,d,g)*(g*(1-pow((1+2*t/pow(g,2)),0.5)) - 1/d);
}

double Acom_d_gamma(double t, double d, double g)
{
    return  Acom(t,d,g)*(1/g + d*(1-pow((1+2*t/pow(g,2)),0.5)) + 2*d*t/pow(g,2)/pow((1+2*t/pow(g,2)),0.5) );
}

double Aunc_d_delta(double t, double d, double g)
{
    return  -Aunc(t,d,g)/d - pow(g,2)*(1-pow((1+2*t/pow(g,2)),0.5))*exp(d*g*(1-pow((1+2*t/pow(g,2)),0.5)))/d;
}

double Aunc_d_gamma(double t, double d, double g)
{
    return  Aunc(t,d,g)/g - g*exp(d*g*(1-pow((1+2*t/pow(g,2)),0.5)))*(1-pow((1+2*t/pow(g,2)),0.5)+ 2*t/pow(g,2)/pow((1+2*t/pow(g,2)),0.5)) ;
}

void c_pair_grad_Poisson_IG_v10(double *Y, double *par, double *hVec, int h, int N, double dt, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, Leb_com, Leb_unc, ddel_Leb_com, dgam_Leb_com, ddel_Leb_unc, dgam_Leb_unc, temp, temp_dnu, temp_ddel, temp_dgam, delta, gamma, nu, Yt, Ys, P1, P2, P3;
    
    nu    = exp(par[0]);
    delta = exp(par[1]);
    gamma = exp(par[2]);
    
    fts[0] = 0; // loglik
    fts[1] = 0; // grad-nu
    fts[2] = 0; // grad-del
    fts[3] = 0; // grad-gam
    for (ih=0; ih<h; ih++){
        hh = (int)hVec[ih];
        
        Leb_com = Acom(hh*dt,delta,gamma);
        Leb_unc = Aunc(hh*dt,delta,gamma);
        
        ddel_Leb_com = Acom_d_delta(hh*dt,delta,gamma);
        dgam_Leb_com = Acom_d_gamma(hh*dt,delta,gamma);
        ddel_Leb_unc = Aunc_d_delta(hh*dt,delta,gamma);
        dgam_Leb_unc = Aunc_d_gamma(hh*dt,delta,gamma);
        
        A1 = nu*Leb_com;
        A2 = nu*Leb_unc;
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            temp_dnu  = 0;
            temp_ddel = 0;
            temp_dgam = 0; 
            for (j=0; j<c+1; j++){
                P1 = pPoi(Yt-j,A2);
                P2 = pPoi(Ys-j,A2);
                P3 = pPoi(j,A1);
                
                temp = temp + P1*P2*P3;
                
                temp_dnu  = temp_dnu  + P1*((Yt-j)/nu - Leb_unc)*P2*P3 + P1*P2*((Ys-j)/nu - Leb_unc)*P3 + P1*P2*P3*(j/nu - Leb_com);
                temp_ddel = temp_ddel + P1*ddel_Leb_unc*((Yt-j)/Leb_unc - nu)*P2*P3 + P1*P2*ddel_Leb_unc*((Ys-j)/Leb_unc - nu)*P3 + P1*P2*P3*ddel_Leb_com*(j/Leb_com - nu);
                temp_dgam = temp_dgam + P1*dgam_Leb_unc*((Yt-j)/Leb_unc - nu)*P2*P3 + P1*P2*dgam_Leb_unc*((Ys-j)/Leb_unc - nu)*P3 + P1*P2*P3*dgam_Leb_com*(j/Leb_com - nu);
            }
            if (temp>0){
                fts[0] = fts[0] + log(temp); 
                fts[1] = fts[1] + temp_dnu/temp; 
                fts[2] = fts[2] + temp_ddel/temp; 
                fts[3] = fts[3] + temp_dgam/temp;
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
   	plhs[0] = mxCreateDoubleMatrix((int)4,1, mxREAL);

	/*  Create a C pointer to a copy of the output matrix. */
	fts = mxGetPr(plhs[0]);
	
    //	printf("Number of rows is %d, number of cols is %d\n",T,T);

    c_pair_grad_Poisson_IG_v10(Y,par,hVec,h,N,dt,fts);
}

