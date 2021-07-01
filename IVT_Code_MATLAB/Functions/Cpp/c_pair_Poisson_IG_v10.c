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



void c_pair_Poisson_IG_v10(double *Y, double *par, double *hVec, int H, int N, double dt, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, temp, delta, gamma, intens, Yt, Ys, SQ;
    
    intens  = exp(par[0]);
    delta   = exp(par[1]);
    gamma   = exp(par[2]);
    
    fts[0] = 0;
    for (ih=0; ih<H; ih++){
        hh = hVec[ih];
        
        A1 = intens*Acom(hh*dt,delta,gamma);
        A2 = intens*Aunc(hh*dt,delta,gamma);
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

    c_pair_Poisson_IG_v10(Y,par,hVec,H,N,dt,fts);
}

