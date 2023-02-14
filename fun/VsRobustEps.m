function Par = VsRobustEps(nr,nc,r,sr,lambda,sp,missrate,tol,options,WEPS,success,times)
% nr,nc,r: matrix nr*nc with rank r
% sr: random start with rank sr
% lambda,sp,missrate: model parameters
% tol,options: glgorithm parameters
% WEPS:iter_eps for PIRNN 
% success: the success tolerance
% times: times experiments  

leps = length(WEPS);
Robust.PIR = zeros(1,leps);
Robust.AIR = 0;
Robust.EPIR = 0;

for itimes = 1:times % 
  B = rand(nr,r); C = rand(r,nc); Y = B * C; Y = Y./max(max(Y));
  % --------------- random mask ---------------
  M_org = zeros(nr,nc); 
  for i=1:nc 
    idx = 1:1:nr; randidx=randperm(nr,nr); % random sequence
    M_org(randidx(1:ceil(nr*missrate)),i)=1; 
  end
  mask = ~M_org; Xm = Y.*mask;
  X0 = rand(nr,nc); [uY,sY,vY] = svd(X0);
  X0 = uY(:,1:sr) * sY(1:sr,1:sr) * vY(:,1:sr)';

  options.Rel = Y;


  optionsP= options;
  for iter_eps = 1:1:leps      
    optionsP.eps = WEPS(iter_eps);
    PIR = ds_ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
    if (PIR.rank(end) == r) && (PIR.RelErr(end) <= success) 
      Robust.PIR(iter_eps) = Robust.PIR(iter_eps) + 1;
    end
  end

  optionsA = options;
  optionsA.eps = 1e0;
  optionsA.mu = 0.75;
  AIR = ds_AdaIRNN(X0,Xm,sp, lambda, mask, tol, optionsA);
  if (AIR.rank(end) == r) && (AIR.RelErr(end) <= success) 
    Robust.AIR = Robust.AIR + 1;
  end

  optionsEP = optionsA;
  optionsEP.alpha = 7e-1;
  EPIR = ds_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsEP); 
  if (EPIR.rank(end) == r) && (EPIR.RelErr(end) <= success) 
    Robust.EPIR = Robust.EPIR + 1;
  end
  Par = Robust;
end