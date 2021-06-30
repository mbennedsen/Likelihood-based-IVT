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

double loggamma(double x)   // Returns log(gamma(x)) for x > 0.
{
    mwIndex j;
    double xx, tmp, y, ser;
    static const double cof[14] = { 57.1562356658629235, -59.5979603554754912, 14.1360979747417471,
                                    -0.491913816097620199, .339946499848118887e-4, .465236289270485756e-4,
                                    -.983744753048795646e-4, .158088703224912494e-3, -.210264441724104883e-3,
                                    .217439618115212643e-3, -.164318106536763890e-3, .844182239838527433e-4,
                                    -.261908384015814087e-4, .368991826595316234e-5};
    
    //if (x <= 0) throw("Bad arg in loggamma");
    y = xx = x;
    tmp = xx + 5.24218750000000000;  // rational 671/128
    tmp = (xx+0.5)*log(tmp)-tmp;
    ser = 0.999999999999997092;
    for (j=0;j<14;j++) ser += cof[j]/++y;
    
    return tmp + log(2.5066282746310005*ser/xx);
}

double sigmoid(double x)
{
    return 1/(1+exp(-x));
}

double pNB(double x, double m, double p) 
{ 
    return exp(loggamma(m+x) - loggamma(m))*pow(1-p,m)*pow(p,x)/fac(x);
}

double Acom(double t, double d, double g)
{
    return (g/d)*exp(d*g*(1-pow((1+2*t/pow(g,2)),0.5) ));
}

double Aunc(double t, double d, double g)
{
    return (g/d)*(1-exp(d*g*(1-pow((1+2*t/pow(g,2)),0.5) )));
}



void c_pair_NB_IG_v10(double *Y, double *par, double *hVec, int H, int N, double dt, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, temp, delta, gamma, m, p, Yt, Ys, SQ;
    
    m     = exp(par[0]);
    p     = sigmoid(par[1]);
    delta = exp(par[2]);
    gamma = exp(par[3]);
    
    fts[0] = 0;
    for (ih=0; ih<H; ih++){
        hh = hVec[ih];
        
        A1 = m*Acom(hh*dt,delta,gamma);
        A2 = m*Aunc(hh*dt,delta,gamma);
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            for (j=0; j<c+1; j++){
                temp = temp + pNB(Yt-j,A2,p)*pNB(Ys-j,A2,p)*pNB(j,A1,p);
            }
            fts[0] = fts[0] + log(temp);
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

    c_pair_NB_IG_v10(Y,par,hVec,H,N,dt,fts);
}

