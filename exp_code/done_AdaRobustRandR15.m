%% AdaIRNN V.S. ProxIRNN with random data  
% this experiment shows the AdaIRNN are more robust than ProxIRNN
% the AdaIRNN has the similar convergence rate with ProxIRNN if the 
% initialization eps are similar
%% --------------- generate data ---------------
clc,clear,format long 
rng(22)
nr = 150; nc = 150; r = 15 ;
% --------------- Synthetic data ---------------
% Y = randn(nr,r) * (randn(r,r)+eye(r,r)) * randn(r,nc);
Y = randn(nr,r) * randn(r,nc);
Y = Y./svds(Y,1);
% Y = randn(nr,r);
% [uY,sY,vY] = svd(Y);
% Y = uY(:,1:r) * sY(1:r,1:r) * vY(:,1:r)';
% Y = Y(:,randperm(nc)); 
clear uY sY vY
% xb = (randn(nr,r) ); xc = randn(r,nc) ;
% Y = xb*xc ; 
% --------------- random mask ---------------
M_org = zeros(nr,nc); 
missrate = 0.5; 
for i=1:nc 
  idx = 1:1:nr;
  randidx=randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~M_org; Xm = Y.*mask;

% %% --------------- parameters ---------------
lambda = 1e-3*norm(Xm,inf);
% lambda = 1e-2;
itmax = 5e3;
sp = 0.5;
tol = 1e-8;
klopt = 1e-5;
weps = 1e-5;
% basic algorithm for sp=0.5 ,SR=0.5
  % Initial point
  % with default eps = eps(1)
%   X0 = zeros(size(Y));
  X0 = randn(nr,r)*randn(r,nc);
%   X0 = Xm;
% %%
  options.Rel = Y;
  options.max_iter = itmax;
  options.KLopt = klopt;
  options.eps = weps;
  options.beta = 1.1;
%   options.teps = weps;

  optionsP= options;
  PIR = ds_ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);

  optionsA = options;
  optionsA.eps = 1e0; 
  optionsA.mu = 0.7; 
  AIR = ds_AdaIRNN(X0,Xm,sp, lambda, mask, tol, optionsA); 

%   optionsEP = optionsA; 
%   optionsEP.alpha = 7e-1; 
%   EPIR = ds_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsEP); 
  %% plot 
pPIR = min(itmax,PIR.iterTol); PIRx = (1:1:pPIR); 
pAIR = min(itmax,AIR.iterTol); AIRx = (1:1:pAIR); 
% pEPI = min(itmax,EPIR.iterTol); EPIRx = (1:1:pEPI); 
% reerr = norm(Y-EPIR.Xsol,"fro") / norm(Y,"fro")

% ------------ rank plot ;
figure(1)
    plot(PIR.rank); hold on
    plot(AIR.rank); 
    legend("ProxIRNN","AdaIRNN")

% ------------ relative error plot
figure(2)
    plot(PIRx,log(PIR.RelErr(1:pPIR)),':k','linewidth',2);hold on
    plot(AIRx,log(AIR.RelErr(1:pAIR)),'--b','linewidth',2);
%     plot(EPIRx,log(EPIR.RelErr(1:pEPI)),'-r','linewidth',2); hold off
    xlabel("iteration"); ylabel("log(RelErr)")
    legend("ProxIRNN","AdaIRNN")

% ------------ relative distance plot 
figure(3)
    plot(PIRx,log(PIR.RelDist(1:pPIR)),':k','linewidth',2);hold on
    plot(AIRx,log(AIR.RelDist(1:pAIR)),'--b','linewidth',2);
%     plot(EPIRx,log(EPIR.RelDist(1:pEPI)),'-r','linewidth',2);hold off
    xlabel("iteration"); ylabel("log(RelDist)")
    legend("ProxIRNN","AdaIRNN")

% ------------ objective value plot 
figure(4)
    plot(PIRx,PIR.f(1:pPIR),':k','linewidth',2);hold on; 
    plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',2);
%     plot(EPIRx,EPIR.f(1:pEPI),'-r','linewidth',2);hold off
    xlabel("iteration"); ylabel("F(x)")
    legend("ProxIRNN","AdaIRNN")  

%%  