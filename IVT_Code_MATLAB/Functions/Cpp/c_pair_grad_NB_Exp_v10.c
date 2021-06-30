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

//void calc_running_sum(double *running_sum, double m, double p, double LebB, int maxInt)
//{
//    running_sum[0] = log(1-p);
//    for (int i = 1; i < maxInt+1; i++)
//    {
//        running_sum[i] = running_sum[i-1] + 1/(LebB*m + i-1);
//    }
//}

void c_pair_grad_NB_Exp_v10(double *Y, double *par, double *hVec, int h, int N, double dt, int maxY, double *running_sum_unc, double *running_sum_com, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, Leb_com, Leb_unc, dlam_Leb_com, dlam_Leb_unc, temp, temp_dm, temp_dp, temp_dlam, lambda, m, p, Yt, Ys, P1, P2, P3;
    
    m    = exp(par[0]);
    p    = sigmoid(par[1]);
    lambda = exp(par[2]);
    
    fts[0] = 0; // loglik
    fts[1] = 0; // grad-m
    fts[2] = 0; // grad-p
    fts[3] = 0; // grad-lam
    for (ih=0; ih<h; ih++){
        hh = (int)hVec[ih];
        
        Leb_com = Acom(hh*dt,lambda);
        Leb_unc = Aunc(hh*dt,lambda);
        
        dlam_Leb_com = Acom_d_lambda(hh*dt,lambda);
        dlam_Leb_unc = Aunc_d_lambda(hh*dt,lambda);
        
        A1 = m*Leb_com;
        A2 = m*Leb_unc;
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            temp_dm  = 0;
            temp_dp  = 0;
            temp_dlam = 0;
            for (j=0; j<c+1; j++){
                
                P1 = pNB(Yt-j,A2,p);
                P2 = pNB(Ys-j,A2,p);
                P3 = pNB(j,A1,p);
                
                temp = temp + P1*P2*P3;
                 
                temp_dm   = temp_dm   + running_sum_unc[(int) ((Yt-j)+ih*(maxY+1))]*P1*P2*P3*Leb_unc + running_sum_unc[(int) ((Ys-j)+ih*(maxY+1))]*P1*P2*P3*Leb_unc + running_sum_com[(int) (j+ih*(maxY+1))]*P1*P2*P3*Leb_com;
                temp_dp   = temp_dp   + ((Yt-j)/p - Leb_unc*m/(1-p))*P1*P2*P3 + ((Ys-j)/p - Leb_unc*m/(1-p))*P1*P2*P3 + (j/p - Leb_com*m/(1-p))*P1*P2*P3;
                temp_dlam = temp_dlam + running_sum_unc[(int) ((Yt-j)+ih*(maxY+1))]*dlam_Leb_unc*m*P1*P2*P3 + running_sum_unc[(int) ((Ys-j)+ih*(maxY+1))]*dlam_Leb_unc*m*P1*P2*P3 + running_sum_com[(int) (j+ih*(maxY+1))]*dlam_Leb_com*m*P1*P2*P3;
            }
            if (temp>0){
                fts[0] = fts[0] + log(temp); 
                fts[1] = fts[1] + temp_dm/temp; 
                fts[2] = fts[2] + temp_dp/temp; 
                fts[3] = fts[3] + temp_dlam/temp; 
            }
        }
    }
}

 
/* The gateway routine */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *par, *hVec, *getH, *getN, *getDT, *Y, *getMaxY, *running_sum_unc, *running_sum_com, *fts;
	int N, h, maxY;
    double dt;

	/*  Check for proper number of arguments. */
	if(nrhs!=9) mexErrMsgTxt("Nine inputs required.");
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
    
    getMaxY        = mxGetPr(prhs[6]);
    maxY           = (int)getMaxY[0];
    
    running_sum_unc        = mxGetPr(prhs[7]);
    running_sum_com        = mxGetPr(prhs[8]);
            
	/*  Set the output pointer to the output matrix. */
   	plhs[0] = mxCreateDoubleMatrix((int)4,1, mxREAL);

	/*  Create a C pointer to a copy of the output matrix. */
	fts = mxGetPr(plhs[0]);
	
    //	printf("Number of rows is %d, number of cols is %d\n",T,T);

    c_pair_grad_NB_Exp_v10(Y,par,hVec,h,N,dt,maxY,running_sum_unc,running_sum_com,fts);
}

