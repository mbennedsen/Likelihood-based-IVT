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



void c_pair_Poisson_GAM_v10(double *Y, double *par, double *hVec, int H, int N, double dt, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, temp, HH, alpha, intens, Yt, Ys, SQ;
    
    intens  = exp(par[0]);
    HH      = exp(par[1]);
    alpha   = exp(par[2]);
    
    fts[0] = 0;
    for (ih=0; ih<H; ih++){
        hh = hVec[ih];
        
        A1 = intens*Acom(hh*dt,HH,alpha);
        A2 = intens*Aunc(hh*dt,HH,alpha);
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            for (j=0; j<c+1; j++){
                temp = temp + pPoi(Yt-j,A2)*pPoi(Ys-j,A2)*pPoi(j,A1);
            }
            fts[0] = fts[0] + log(temp); ///(N-hh);  
        }
    }
}

 
/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *par, *getH, *hVec, *getN, *getDT, *Y, *fts;
	int N, H, h;
    double dt;

	/*  Check for proper number of arguments. */
	if(nrhs!=6) mexErrMsgTxt("Six inputs required.");
	if(nlhs!=1) mexErrMsgTxt("One output required."); 

	/*  Create a pointer to the input matrices . */

    Y           = mxGetPr(prhs[0]);
    par         = mxGetPr(prhs[1]);
    hVec        = mxGetPr(prhs[2]);
    getH        = mxGetPr(prhs[3]);
    H           = (int)getH[0];
	getN        = mxGetPr(prhs[4]);
    N           = (int)getN[0];
	getDT       = mxGetPr(prhs[5]);
    dt          = (double)getDT[0];
            
	/*  Set the output pointer to the output matrix. */
   	plhs[0] = mxCreateDoubleMatrix((int)1,1, mxREAL);

	/*  Create a C pointer to a copy of the output matrix. */
	fts = mxGetPr(plhs[0]);
	
    //	printf("Number of rows is %d, number of cols is %d\n",T,T);

    c_pair_Poisson_GAM_v10(Y,par,hVec,H,N,dt,fts);
}

