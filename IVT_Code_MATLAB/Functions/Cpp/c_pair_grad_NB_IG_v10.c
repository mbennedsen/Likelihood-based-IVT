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

void c_pair_grad_NB_IG_v10(double *Y, double *par, double *hVec, int h, int N, double dt, int maxY, double *running_sum_unc, double *running_sum_com, double *fts)
{
 	mwIndex i,j,ih; 
    int c, hh;
    double A1, A2, Leb_com, Leb_unc, ddel_Leb_com, dgam_Leb_com, ddel_Leb_unc, dgam_Leb_unc, temp, temp_dm, temp_dp, temp_ddel, temp_dgam, delta, gamma, m, p, Yt, Ys, P1, P2, P3;
    
    m    = exp(par[0]);
    p    = sigmoid(par[1]);
    delta = exp(par[2]);
    gamma = exp(par[3]);
    
    fts[0] = 0; // loglik
    fts[1] = 0; // grad-m
    fts[2] = 0; // grad-p
    fts[3] = 0; // grad-del
    fts[4] = 0; // grad-gam
    for (ih=0; ih<h; ih++){
        hh = (int)hVec[ih];
        
        Leb_com = Acom(hh*dt,delta,gamma);
        Leb_unc = Aunc(hh*dt,delta,gamma);
        
        ddel_Leb_com = Acom_d_delta(hh*dt,delta,gamma);
        dgam_Leb_com = Acom_d_gamma(hh*dt,delta,gamma);
        ddel_Leb_unc = Aunc_d_delta(hh*dt,delta,gamma);
        dgam_Leb_unc = Aunc_d_gamma(hh*dt,delta,gamma);
        
        A1 = m*Leb_com;
        A2 = m*Leb_unc;
        for (i=0; i<(N-hh); i++) {

            Yt = Y[i+hh];
            Ys = Y[i];

            c  = (int)fmin(Yt,Ys);
            temp = 0;
            temp_dm  = 0;
            temp_dp  = 0;
            temp_ddel = 0;
            temp_dgam = 0; 
            for (j=0; j<c+1; j++){
                P1 = pNB(Yt-j,A2,p);
                P2 = pNB(Ys-j,A2,p);
                P3 = pNB(j,A1,p);
                
                temp = temp + P1*P2*P3;
                
                temp_dm   = temp_dm   + running_sum_unc[(int) ((Yt-j)+ih*(maxY+1))]*P1*P2*P3*Leb_unc + running_sum_unc[(int) ((Ys-j)+ih*(maxY+1))]*P1*P2*P3*Leb_unc + running_sum_com[(int) (j+ih*(maxY+1))]*P1*P2*P3*Leb_com;
                temp_dp   = temp_dp   + ((Yt-j)/p - Leb_unc*m/(1-p))*P1*P2*P3 + ((Ys-j)/p - Leb_unc*m/(1-p))*P1*P2*P3 + (j/p - Leb_com*m/(1-p))*P1*P2*P3;
                
                temp_ddel = temp_ddel + running_sum_unc[(int) ((Yt-j)+ih*(maxY+1))]*ddel_Leb_unc*m*P1*P2*P3 + running_sum_unc[(int) ((Ys-j)+ih*(maxY+1))]*ddel_Leb_unc*m*P1*P2*P3 + running_sum_com[(int) (j+ih*(maxY+1))]*ddel_Leb_com*m*P1*P2*P3;
                temp_dgam = temp_dgam + running_sum_unc[(int) ((Yt-j)+ih*(maxY+1))]*dgam_Leb_unc*m*P1*P2*P3 + running_sum_unc[(int) ((Ys-j)+ih*(maxY+1))]*dgam_Leb_unc*m*P1*P2*P3 + running_sum_com[(int) (j+ih*(maxY+1))]*dgam_Leb_com*m*P1*P2*P3;
            }
            if (temp>0){
                fts[0] = fts[0] + log(temp); 
                fts[1] = fts[1] + temp_dm/temp; 
                fts[2] = fts[2] + temp_dp/temp; 
                fts[3] = fts[3] + temp_ddel/temp; 
                fts[4] = fts[4] + temp_dgam/temp;
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
   	plhs[0] = mxCreateDoubleMatrix((int)5,1, mxREAL);

	/*  Create a C pointer to a copy of the output matrix. */
	fts = mxGetPr(plhs[0]);
	
    //	printf("Number of rows is %d, number of cols is %d\n",T,T);

    c_pair_grad_NB_IG_v10(Y,par,hVec,h,N,dt,maxY,running_sum_unc,running_sum_com,fts);
}

