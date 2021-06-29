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

double sigmoid(double x)
{
    return 1.0/(1.0 + exp(-x));
}

double pPoi(double y, double lam) 
{ 
    return pow(lam,y)*exp(-lam)/fac(y);
}

double Acom(double t, double H, double a)
{
    return a*pow(1+t/a,-H)/H;
}

double Aunc(double t, double H, double a)
{
    return a*(1-pow(1+t/a,-H))/H;
}

double Acom_d_H(double t, double H, double a)
{
    return  -a*pow(1+t/a,-H)*(1/H + log(1+t/a))/H;
}

double Acom_d_a(double t, double H, double a)
{
    return  pow(1+t/a,-H-1)*( (1+t/a)/H + t/a );
}

double Aunc_d_H(double t, double H, double a)
{
    return  -a/H/H - Acom_d_H(t,H,a);
}

double Aunc_d_a(double t, double H, double a)
{
    return  1/H - Acom_d_a(t,H,a);
}

void c_pair_grad_Poisson_GAM_v10(double *Y, double *par, double *hVec, int h, int N, double dt, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, Leb_com, Leb_unc, dH_Leb_com, da_Leb_com, dH_Leb_unc, da_Leb_unc, temp, temp_dnu, temp_dH, temp_da, HH, alpha, nu, Yt, Ys, P1, P2, P3;
    
    nu      = exp(par[0]);
    HH      = exp(par[1]);
    alpha   = exp(par[2]);
    
    fts[0] = 0; // loglik
    fts[1] = 0; // grad-nu
    fts[2] = 0; // grad-H
    fts[3] = 0; // grad-alp
    for (ih=0; ih<h; ih++){
        hh = (int)hVec[ih];
        
        Leb_com = Acom(hh*dt,HH,alpha);
        Leb_unc = Aunc(hh*dt,HH,alpha);
        
        dH_Leb_com = Acom_d_H(hh*dt,HH,alpha);
        da_Leb_com = Acom_d_a(hh*dt,HH,alpha);
        dH_Leb_unc = Aunc_d_H(hh*dt,HH,alpha);
        da_Leb_unc = Aunc_d_a(hh*dt,HH,alpha);
        
        A1 = nu*Leb_com;
        A2 = nu*Leb_unc;
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            temp_dnu = 0;
            temp_dH  = 0;
            temp_da  = 0;
            for (j=0; j<c+1; j++){
                P1 = pPoi(Yt-j,A2);
                P2 = pPoi(Ys-j,A2);
                P3 = pPoi(j,A1);
                
                temp = temp + P1*P2*P3;
                
                temp_dnu = temp_dnu  + P1*((Yt-j)/nu - Leb_unc)*P2*P3 + P1*P2*((Ys-j)/nu - Leb_unc)*P3 + P1*P2*P3*(j/nu - Leb_com);
                temp_dH  = temp_dH + P1*dH_Leb_unc*((Yt-j)/Leb_unc - nu)*P2*P3 + P1*P2*dH_Leb_unc*((Ys-j)/Leb_unc - nu)*P3 + P1*P2*P3*dH_Leb_com*(j/Leb_com - nu);
                temp_da  = temp_da + P1*da_Leb_unc*((Yt-j)/Leb_unc - nu)*P2*P3 + P1*P2*da_Leb_unc*((Ys-j)/Leb_unc - nu)*P3 + P1*P2*P3*da_Leb_com*(j/Leb_com - nu);
            }
            if (temp>0){
                fts[0] = fts[0] + log(temp); 
                fts[1] = fts[1] + temp_dnu/temp; 
                fts[2] = fts[2] + temp_dH/temp; 
                fts[3] = fts[3] + temp_da/temp;
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

    c_pair_grad_Poisson_GAM_v10(Y,par,hVec,h,N,dt,fts);
}

