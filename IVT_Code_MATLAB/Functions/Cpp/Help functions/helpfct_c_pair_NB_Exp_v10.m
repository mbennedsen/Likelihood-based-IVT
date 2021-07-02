function [loglik,grad] = helpfct_c_pair_NB_Exp_v10(y,par,Hvec,H,n,dt)

m = exp(par(1));
p = sigmoid(par(2));
l = exp(par(3));

maxY = max(y);

LebUnc = (1-exp(-l*dt*Hvec))/l;
LebCom = exp(-l*dt*Hvec)/l;

if length(Hvec) == 2
    r_sum_unc = [cumsum([log(1-p),1./(LebUnc(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(2)*m + (0:(maxY-1))) ])];
    r_sum_com = [cumsum([log(1-p),1./(LebCom(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(2)*m + (0:(maxY-1))) ])];
elseif length(Hvec) == 3
    r_sum_unc = [cumsum([log(1-p),1./(LebUnc(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(2)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(3)*m + (0:(maxY-1))) ])];
    r_sum_com = [cumsum([log(1-p),1./(LebCom(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(2)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(3)*m + (0:(maxY-1))) ])];
elseif length(Hvec) == 4
    r_sum_unc = [cumsum([log(1-p),1./(LebUnc(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(2)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(3)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(4)*m + (0:(maxY-1))) ])];
    r_sum_com = [cumsum([log(1-p),1./(LebCom(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(2)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(3)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(4)*m + (0:(maxY-1))) ])];    
elseif length(Hvec) == 1
    r_sum_unc = cumsum([log(1-p),1./(LebUnc(1)*m + (0:(maxY-1))) ]);
    r_sum_com = cumsum([log(1-p),1./(LebCom(1)*m + (0:(maxY-1))) ]);
else
    r_sum_unc = [cumsum([log(1-p),1./(LebUnc(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(2)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(3)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebUnc(4)*m + (0:(maxY-1))) ])];
    r_sum_com = [cumsum([log(1-p),1./(LebCom(1)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(2)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(3)*m + (0:(maxY-1))) ]),cumsum([log(1-p),1./(LebCom(4)*m + (0:(maxY-1))) ])];    

    for i = 5:length(Hvec)
         r_sum_unc = [r_sum_unc,cumsum([log(1-p),1./(LebUnc(i)*m + (0:(maxY-1))) ])];
         r_sum_com = [r_sum_com,cumsum([log(1-p),1./(LebCom(i)*m + (0:(maxY-1))) ])];
    end
end
%r_sum_unc(0 + 1)
%r_sum_unc(maxY + 2)

output = c_pair_grad_NB_Exp_v10(y,par,Hvec,H,n,dt,maxY,r_sum_unc,r_sum_com);

loglik = -1*output(1);
grad   = -1*[m*output(2);
             p^2*exp(-par(2))*output(3);    
             l*output(4)];