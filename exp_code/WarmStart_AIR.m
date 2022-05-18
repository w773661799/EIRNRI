

%% ? warm start 
% warm start with AIRNN to a 
sp = 0.3; 
lambdaRho = 0.3 ; 

X_INIT_WS = XM;
  for i=1:3
    Xm = XM(:,:,i); 
    lambda = 100;
    optionsA.max_iter = 5e1;
    X_INIT_AIR = X_INIT_RAND1; 
    while lambda>norm(Xm,"fro")*1e-1
      AIR = MC_AIRNN(X_INIT_AIR,Xm,sp, lambda, mask, tol, optionsA);
      X_INIT_AIR = AIR.Xsol;
      lambda = lambda*lambdaRho;
%       optionsA.eps = AIR.weps;
    end
    X_INIT_WS(:,:,i) = AIR.Xsol;
  end % end for warm start
 % strat from the X_INIT_WS for different methods 
 % compare the solution of IRNN/ PIRNN/ AIRNN/ EPIRNN/ SCP+ADMM/ FGSRp+ADMM
%
options.max_iter = 1e3;
 for i= 1:3
  Xm = XM(:,:,i); lambda = norm(Xm,"fro")*1e-5;
  
  optionsP = options; optionsP.eps = 1e-5;
  PIR = MC_PIRNN(X_INIT_WS(:,:,i),Xm,sp, lambda, mask, tol, optionsP);
  X_PIR(:,:,i) = PIR.Xsol;
  
  optionsA = options; optionsA.eps = 1; optionsA.Scalar = 0.5;
  AIR = MC_AIRNN(X_INIT_WS(:,:,i),Xm,sp, lambda, mask, tol, optionsA);
  X_AIR(:,:,i) = AIR.Xsol;
  
  optionsEP = options; optionsEP.eps = 1; optionsEP.Scalar = 0.5;
  optionsEP.alpha = 0.7;
  EPIR = MC_EPIRNN(X_INIT_WS(:,:,i),Xm,sp, lambda, mask, tol, optionsEP);
  X_EPIR(:,:,i) = EPIR.Xsol;
end
 % % ----------------------------------------------------------------------- 
 
 %%