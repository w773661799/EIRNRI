%% sensitive of PIRNN / AIRNN /EPIRNN with eps 
clc,clear,format long 
rng(22)
nr = 150; nc = 150; r = 15 ;
% --------------- Synthetic data ---------------
xb = 1+randn(nr,r); xc = 2+randn(r,nc) ;
Y = xb*xc ; 
% --------------- random mask ---------------
M_org = zeros(nr,nc); 
missrate = 0.5; 
for i=1:nc  
    idx = 1:1:nr;
    randidx=randperm(nr,nr); % random sequence
    M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~M_org; 
Xm=Y.*mask; 
% --------------- parameters ---------------
orieps_spl = [1,1e-2,1e-4,1e-6] ;
lambda = 1e-3*norm(Y,inf);
tol_spl = [1e-6; 1e-7; 1e-8]; 
itmax = 5e3; 
  %% basic algorithm for sp=0.1 ,SR=0.5
  % Initial point 
  rcSen = 10; X0 = (randn(nr,rcSen))*randn(rcSen,nc); 
  sp = 0.5; 
  optionsP.Rel = Y; 
  optionsP.max_iter = itmax;
  optionsP.Scalar = 0.3; 
  optionsP.alpha = 0.7; 
  optionsP.KLopt = 1e-5;
  tol = 5e-6; 
  for i = 1:length(orieps_spl)
    optionsP.eps = orieps_spl(i);  
    PIReps{i} = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
    AIReps{i} = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
    EPIReps{i} = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
  end

  %%
  Titor = []; Trank = []; Tobj=[];
  for i = 1:length(orieps_spl)
    Titor(i,1:3) = [PIReps{i}.iterTol, AIReps{i}.iterTol, EPIReps{i}.iterTol];
    Trank(i,1:3) = [PIReps{i}.rank(end), AIReps{i}.rank(end), EPIReps{i}.rank(end)];
    Tobj(i,1:3) = [PIReps{i}.f(end), AIReps{i}.f(end), EPIReps{i}.f(end)];
  end
  
 %% plot 
 pit = 1;
    pPIR = min(itmax,PIReps{pit}.iterTol); pAIR = min(itmax,AIReps{pit}.iterTol);
    pEPI = min(itmax,EPIReps{pit}.iterTol); 
    PIRx = (1:1:pPIR); AIRx = (1:1:pAIR); EPIRx = (1:1:pEPI); 
%     % ------------ RelErr plot 
%     figure(1) % RelErr
%     plot(PIRx,log10(PIReps{pit}.RelErr(1:pPIR)),':.r','linewidth',1);hold on
%     plot(AIRx,log10(AIReps{pit}.RelErr(1:pAIR)),'--b','linewidth',1);
%     plot(EPIRx,log10(EPIReps{pit}.RelErr(1:pEPI)),'-.g','linewidth',1);hold off
%     xlabel("iteration"); ylabel("log(RelErr)")
%     legend("PIRNN","AIRNN","EPIRNN")
    % ------------ RelDist plot 
figure(2) % Errdist 
    plot(PIRx,log10(PIReps{pit}.RelDist(1:pPIR)),':.r','linewidth',2);hold on
    plot(AIRx,log10(AIReps{pit}.RelDist(1:pAIR)),'--b','linewidth',2);
    plot(EPIRx,log10(EPIReps{pit}.RelDist(1:pEPI)),'-.g','linewidth',2);hold off
    xlabel("iteration"); ylabel("log(RelDist)")
    legend("PIRNN","AIRNN","EPIRNN")

figure(3) % obj 
    plot(PIRx,PIReps{pit}.f(1:pPIR),':.r','linewidth',2);hold on
    plot(AIRx,AIReps{pit}.f(1:pAIR),'--b','linewidth',2);
    plot(EPIRx,EPIReps{pit}.f(1:pEPI),'-.g','linewidth',2);hold off
    xlabel("iteration"); ylabel("F(X)")
    legend("PIRNN","AIRNN","EPIRNN")

figure(4) %  rank  
  plot(PIRx,PIReps{pit}.rank,':.r','linewidth',2);hold on; 
  plot(AIRx,AIReps{pit}.rank,'--b','linewidth',2);
  plot(EPIRx,EPIReps{pit}.rank,'-.g','linewidth',2);hold off 
%   title("rank of iterations")
  xlabel("iteration"); ylabel("rank")
  legend("PIRNN","AIRNN","EPIRNN")   