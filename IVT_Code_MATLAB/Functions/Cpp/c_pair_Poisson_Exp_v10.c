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
    return 1/(1+exp(-x));
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


void c_pair_Poisson_Exp_v10(double *Y, double *par, double *hVec, int H, int N, double dt, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, LebA, temp, lambda, intens, Yt, Ys;
    
    intens  = exp(par[0]);
    lambda  = exp(par[1]);
    
    LebA = 1/lambda;
    
    fts[0] = 0;
    for (ih=0; ih<H; ih++){
        hh = hVec[ih];
        
        A1 = intens*Acom(hh*dt,lambda);
        A2 = intens*Aunc(hh*dt,lambda);
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            for (j=0; j<c+1; j++){
                temp = temp + pPoi(Yt-j,A2)*pPoi(Ys-j,A2)*pPoi(j,A1);
            }
            fts[0] = fts[0] + log(temp); ///(N-h);

        }
    }
}

 
/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *par, *getH, *hVec, *getN, *getDT, *Y, *fts;
	int N, H;
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

    c_pair_Poisson_Exp_v10(Y,par,hVec,H,N,dt,fts);
}

