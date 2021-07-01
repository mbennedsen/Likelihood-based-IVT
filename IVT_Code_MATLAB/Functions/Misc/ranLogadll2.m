function  ret = ranLogadll2(p,n,m)

  v = random('Uniform',0,1,n,m);
  u = random('Uniform',0,1,n,m);      

  ret = floor(1+ log(v)./log(1-(1-p).^u));

