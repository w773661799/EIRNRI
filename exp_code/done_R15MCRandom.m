%% test the ds_ProxIRNN / AIRNN / EIRNRI with random data
%% Experiment shows the EIRNRI will convergence faster than PIRNN
% the AIRNN has the similar convergence rate with ds_ProxIRNN
% --------------- generate data ---------------
clc,clear,format long; rng(23)
nr = 150; nc = 150; r = 15;
% --------------- Synthetic data ---------------
Y = randn(nr,r) * randn(r,nc);
% --------------- random mask ---------------
M_org = zeros(nr,nc); missrate = 0.5; 
for i=1:nc 
  randidx=randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end

mask = ~M_org; Xm = Y.*mask;

% %% --------------- parameters ---------------
lambda = 1e-1*norm(Xm,inf);
itmax = 1e3; 
sp = 0.5; 
tol = 1e-7; 
klopt = 1e-5;
seps = 1e-3;
% weps = 1e-16;
options.Rel = Y; 
options.max_iter = itmax; 
options.KLopt = klopt;
options.eps = seps; 
options.beta = 1.1; 
% options.teps = weps;
% r=5;
% Initial point
% X0 = randn(nr,r)*randn(r,nc); 
  optionsP= options;
  PIR = ds_ProxIRNN(Xm,Xm,sp, lambda, mask, tol, optionsP);

  optionsAC = options;
  ACC = AIRNN_2021(Xm,Xm,sp, lambda, mask, tol, optionsAC);

  optionsA = options; 
%   optionsA.eps = 1e-3;
  optionsA.mu = 0.1;
  AIR = ds_AdaIRNN(Xm,Xm,sp, lambda, mask, tol, optionsA); 

  optionsEP = optionsA; 
  optionsEP.alpha = 7e-1; 
  EPIR = ds_EPIRNN(Xm,Xm,sp, lambda, mask, tol, optionsEP); 

% plot 
  pPIR = min(itmax,PIR.iterTol); PIRx = (1:1:pPIR);
  pPAC = min(itmax,ACC.iterTol); PACx = (1:1:pPAC);
  pAIR = min(itmax,AIR.iterTol); AIRx = (1:1:pAIR);
  pEPI = min(itmax,EPIR.iterTol); EPIRx = (1:1:pEPI);
%%
figure(1)
% ------------ relative error plot
  plot(PIRx,log10(PIR.RelErr(1:pPIR)),':k','linewidth',2);hold on
  plot(PACx,log10(ACC.RelErr(1:pPAC)),'--b','linewidth',2);
  
  plot(AIRx,log10(AIR.RelErr(1:pAIR)),'--b','linewidth',2);
  plot(EPIRx,log10(EPIR.RelErr(1:pEPI)),'-r','linewidth',2); hold off
  xlabel("iteration"); ylabel("log_{10}(RelErr)")
  legend("PIRNN","IRNRI","EIRNRI")
%%
figure(2)
% ------------ relative distance plot 
  plot(PIRx,log10(PIR.RelDist(1:pPIR)),':k','linewidth',2);hold on
  plot(PIRx,log10(PIR.RelDist(1:pPIR)),':k','linewidth',2);hold on
  plot(AIRx,log10(AIR.RelDist(1:pAIR)),'--b','linewidth',2);
  plot(EPIRx,log10(EPIR.RelDist(1:pEPI)),'-r','linewidth',2);hold off
  xlabel("iteration"); ylabel("log_{10}(RelDist)")
  legend("PIRNN","IRNRI","EIRNRI")
%%
figure(3)
% ------------ objective value plot 
  plot(PIRx,PIR.f(1:pPIR),':k','linewidth',2);hold on; 
  

  plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',2);

  plot(EPIRx,EPIR.f(1:pEPI),'-r','linewidth',2);hold off
  xlabel("iteration"); ylabel("F(x)")
  legend("PIRNN","IRNRI","EIRNRI")
%% % subplot
h = figure(1);
  set(h,'Position',[100 100 1500 500]);
subplot('Position',[0.05,0.1,0.28,0.85])
  plot(PIRx,log10(PIR.RelErr(1:pPIR)),':k','linewidth',2);hold on
  plot(AIRx,log10(AIR.RelErr(1:pAIR)),'--b','linewidth',2);
  plot(EPIRx,log10(EPIR.RelErr(1:pEPI)),'-.r','linewidth',2); hold off
  xlabel("iteration"); ylabel("log_{10}(RelErr)")
  legend("PIRNN","AIRNN","EIRNRI")
subplot('Position',[0.38,0.1,0.28,0.85])
  plot(PIRx,log10(PIR.RelDist(1:pPIR)),':k','linewidth',2);hold on
  plot(AIRx,log10(AIR.RelDist(1:pAIR)),'--b','linewidth',2);
  plot(EPIRx,log10(EPIR.RelDist(1:pEPI)),'-r','linewidth',2);hold off
  xlabel("iteration"); ylabel("log_{10}(RelDist)")
  legend("PIRNN","AIRNN","EIRNRI")
subplot('Position',[0.71,0.1,0.28,0.85])
  plot(PIRx,PIR.f(1:pPIR),':k','linewidth',2);hold on; 
  plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',2);
  plot(EPIRx,EPIR.f(1:pEPI),'-r','linewidth',2);hold off
  xlabel("iteration"); ylabel("F(x)")
  legend("PIRNN","AIRNN","EIRNRI")
%% -------------------- sensitive of alpha --------------------
% sensitive of alpha for extrapolation parameter 
clc; clear optionsEP
optionsEP = options;
Lalpha = [0 0.1 0.3 0.5 0.7 0.9];
for i = 1:length(Lalpha)
  optionsEP.alpha = Lalpha(i) ;
  LEPIR{i} = ds_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsEP);
end
%% plot 
%    set(h,'Position',[500 500 1500 500]);
figure(1) % RelErr
set (gcf,'position',[150 50 600 400] );
% set (gca,'position',[0.08,0.1,0.9,0.9]);
% set(gca,'looseInset',[0 0 0 0])
  pltLEPx = min(itmax,LEPIR{i}.iterTol);
  plot(log10(LEPIR{1}.RelErr),':dg','linewidth',1);hold on
  plot(log10(LEPIR{2}.RelErr),'--b','linewidth',2);
  plot(log10(LEPIR{3}.RelErr),'-m','linewidth',2);
  plot(log10(LEPIR{4}.RelErr),'-.k','linewidth',2);
  plot(log10(LEPIR{5}.RelErr),':>c','linewidth',1);
  plot(log10(LEPIR{6}.RelErr),'-r','linewidth',2);hold off
  xlabel("iteration"); ylabel("log_{10}(RelErr)")
  legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
    "\alpha = 0.7","\alpha = 0.9")
% set (gca,'fontsize',12);

figure(2) % Errdist
set (gcf,'position',[50 50 600 600] );
% set (gca,'position',[0.08,0.08,0.9,0.9] );
  plot(log10(LEPIR{1}.RelDist),':dg','linewidth',1);hold on
  plot(log10(LEPIR{2}.RelDist),'--b','linewidth',2);
  plot(log10(LEPIR{3}.RelDist),'-m','linewidth',2);
  plot(log10(LEPIR{4}.RelDist),'-.k','linewidth',2);
  plot(log10(LEPIR{5}.RelDist),':>c','linewidth',1);
  plot(log10(LEPIR{6}.RelDist),'-r','linewidth',2);hold off
  xlabel("iteration"); ylabel("log_{10}(RelDist)")
  legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
  "\alpha = 0.7","\alpha = 0.9")
% set (gca,'fontsize',18);

figure(3) % obj
set (gcf,'position',[100 10 600 400] );
% set (gca,'position',[0.08,0.08,0.9,0.9] );
  plot(log10(LEPIR{1}.f),':dg','linewidth',1);hold on
  plot(log10(LEPIR{2}.f),'--b','linewidth',2);
  plot(log10(LEPIR{3}.f),'-m','linewidth',2);
  plot(log10(LEPIR{4}.f),'-.k','linewidth',2);
  plot(log10(LEPIR{5}.f),':>c','linewidth',1);
  plot(log10(LEPIR{6}.f),'-r','linewidth',1);hold off
  xlabel("iteration"); ylabel("F(X)")
  legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
  "\alpha = 0.7","\alpha = 0.9")
% set (gca,'fontsize',18);
% -/end -------------------------------------------------------------------
%% --------------------------------------------------------------



%% Test the unsuceeful tol %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc,clear,format long; rng(23); 
% p = parpool(16); 
nr = 150; nc = 150; 
% %% --------------- parameters ---------------
lambda = 1e-2;
itmax = 5e3;
sp = 0.5;
tol = 1e-7;
klopt = 1e-5;
beta = 1.1;
success = 1e-2;
% ------------------------------------------------------------------------
missrate = 0.5; 
weps = 1e-4;
times = 20;
init_rank_max = 5;
WEPS = [1e-2, 1e-3, 5e-4, 1e-4];
options.max_iter = itmax;
options.KLopt = klopt;
%   options.eps = weps;
options.beta = beta;

% -------------------------- 75 *20 times --------------------------
% with different initialization rank: 0--74
% for each initialization rank we test 20 times
Rank = [25,30];

Robust.PIR = zeros(length(Rank),length(WEPS));
Robust.AIR = zeros(size(Rank));
Robust.EPIR = zeros(size(Rank));

options.max_iter = itmax;
options.KLopt = klopt;
%       options.eps = weps;
options.beta = beta;

irank = 1; %3
r = Rank(2);
sr = 50;% 74
leps = length(WEPS);
Robust.PIR = zeros(1,leps);
Robust.AIR = 0;
Robust.EPIR = 0;
%%
for itimes = 1:100 % 
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
%   for iter_eps = 1:1:leps      
%     optionsP.eps = WEPS(iter_eps);
%     PIR = ds_ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
%     if (PIR.rank(end) == r) && (PIR.RelErr(end) <= success) 
%       Robust.PIR(iter_eps) = Robust.PIR(iter_eps) + 1;
%     end
%   end

  optionsA = options;
  optionsA.eps = 1e0;
  optionsA.mu = 0.95;
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
end
% par{r_iter} = VsRobustEps(nr,nc,r,r_iter,lambda,sp,missrate,tol,options,WEPS,success,1);
% -------------------------------------
%%
Robust.PIR(irank,:) = Robust.PIR(irank,:) + par{iter}.PIR;
Robust.AIR(irank) = Robust.AIR(irank) + par{iter}.AIR;
Robust.EPIR(irank) = Robust.EPIR(irank) + par{iter}.EPIR;



  %%
  clc;
  PIR.RelErr(end)
  AIR.RelErr(end)
  EPIR.RelErr(end)
%   reerr = norm(Y-EPIR.Xsol,"fro") / norm(Y,"fro")
%   reerr = norm(mask.*(Y-EPIR.Xsol),"fro") / norm(mask.*(Y),"fro")
  %% 
  plot(PIR.rank); hold on
  plot(AIR.rank)
  plot(EPIR.rank);hold off
  legend("PIR","AIR","EP")
    %% plot
    pPIR = min(itmax,PIR.iterTol); pAIR = min(itmax,AIR.iterTol);
    pEPI = min(itmax,EPIR.iterTol); 
    PIRx = (1:1:pPIR); AIRx = (1:1:pAIR); EPIRx = (1:1:pEPI); 
    %%% plot subplot 
% ------------ RelErr plot 
h = figure(1);
%     set (gca,'position',[0.1,0.1,0.9,0.9] );
    set(h,'Position',[200 300 1500 500]);
%     subplot(1,3,1)
subplot('Position',[0.05,0.1,0.28,0.85])
    plot(PIRx,log(PIR.RelErr(1:pPIR)),':k','linewidth',2);hold on
    plot(AIRx,log(AIR.RelErr(1:pAIR)),'--b','linewidth',2);
    plot(EPIRx,log(EPIR.RelErr(1:pEPI)),'-.r','linewidth',2); hold off
    xlabel("iteration"); ylabel("log(RelErr)")
    legend("PIRNN","AIRNN","EIRNRI")
subplot('Position',[0.38,0.1,0.28,0.85])
%     subplot(1,3,2,'position',[0.35,0,0.3,1])
    plot(PIRx,log(PIR.RelDist(1:pPIR)),':k','linewidth',2);hold on
    plot(AIRx,log(AIR.RelDist(1:pAIR)),'--b','linewidth',2);
    plot(EPIRx,log(EPIR.RelDist(1:pEPI)),'-r','linewidth',2);hold off
    xlabel("iteration"); ylabel("log(RelDist)")
    legend("PIRNN","AIRNN","EIRNRI")
subplot('Position',[0.71,0.1,0.28,0.85])
%     subplot(1,3,3,'position',[0.7,0,0.3,1])
    plot(PIRx,PIR.f(1:pPIR),':k','linewidth',2);hold on; 
    plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',2);
    plot(EPIRx,EPIR.f(1:pEPI),'-r','linewidth',2);hold off
    xlabel("iteration"); ylabel("F(x)")
    legend("PIRNN","AIRNN","EIRNRI")
    %% plot 3 
  figure(1)
    plot(PIRx,log(PIR.RelErr(1:pPIR)),':k','linewidth',2);hold on
    plot(AIRx,log(AIR.RelErr(1:pAIR)),'--b','linewidth',2);
    plot(EPIRx,log(EPIR.RelErr(1:pEPI)),'-r','linewidth',2); hold off
    xlabel("iteration"); ylabel("log(RelErr)")
    legend("PIRNN","AIRNN","EIRNRI")
    % ------------ RelDist plot 
  figure(2)
    plot(PIRx,log(PIR.RelDist(1:pPIR)),':k','linewidth',2);hold on
    plot(AIRx,log(AIR.RelDist(1:pAIR)),'--b','linewidth',2);
    plot(EPIRx,log(EPIR.RelDist(1:pEPI)),'-r','linewidth',2);hold off
    xlabel("iteration"); ylabel("log(RelDist)")
    legend("PIRNN","AIRNN","EIRNRI")
    % ------------ objective value plot 
  figure(3)
    plot(PIRx,PIR.f(1:pPIR),':k','linewidth',2);hold on; 
    plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',2);
    plot(EPIRx,EPIR.f(1:pEPI),'-r','linewidth',2);hold off
    xlabel("iteration"); ylabel("F(x)")
    legend("PIRNN","AIRNN","EIRNRI")
% -/end -----------------------------------------------------------------
%-----------------------------------------------------------------
%% robust of the initial points and eps for AIRNN
% Initial point 
clc
rcSen = 15; 
X0 = randn(nr,nc); 
% X0 = (randn(nr,rcSen) + 1)*randn(rcSen,nc); 
lambda = 1e-3*norm(Y,inf);
itmax = 5e3; 
sp = 0.5; 
options.Rel = Y; 
options.beta = 1.3;
options.max_iter = itmax;
options.mu = 0.1; 
options.alpha = 0.7; 
options.KLopt = 1e-5;
%
  orieps_spl = 10.^(0:-1:-5);
  optionsA = options;
  optionsA.eps = 1e0;
%     optionsA.zero = 1e-4;
  tol = 1e-6; 
  for i = 1:length(orieps_spl)
    optionsP = options;
    optionsP.eps = orieps_spl(i);  
    PIReps{i} = ds_ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
%     optionsA.eps = orieps_spl(i);
  end
    AIReps{1} = ds_AdaIRNN(X0,Xm,sp, lambda, mask, tol, optionsA); 
    EPIReps{1} = ds_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsA); 
%     optionsA.eps = orieps_spl(1);
%     AIReps{1} = ds_AdaIRNN(X0,Xm,sp, lambda, mask, tol, optionsA); 
%     EPIReps{1} = ds_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsA); 
  %%
  clear F R T
  
  for i = 1:6
    F(i,1:3) = [PIReps{i}.f(end), AIReps{i}.f(end),EPIReps{i}.f(end)];
    R(i,1:3) = [PIReps{i}.rank(end), AIReps{i}.rank(end),EPIReps{i}.rank(end)];
    T(i,1:3) = [PIReps{i}.time(end), AIReps{i}.time(end),EPIReps{i}.time(end)];
  end
format short g
  
%% polt sensitive of eps for ds_ProxIRNN and robust for AIRNN/EIRNRI
%     timeStveps 
% time VS f
figure(1)

stimeeps = 0.03;
timelen = 0.97;

% for epsidx =1:length(orieps_spl)
for epsidx =2:2:4
  begtime = find(PIReps{epsidx}.time>stimeeps ,1); 
  endtime = find(PIReps{epsidx}.time>stimeeps+timelen ,1);
  plot(PIReps{epsidx}.time(begtime:endtime), ...
    PIReps{epsidx}.f(begtime:endtime),".",'linewidth',2); hold on
  
  begtime = find(AIReps{epsidx}.time>stimeeps ,1);
  endtime = find(AIReps{epsidx}.time>stimeeps+timelen ,1);
  plot(AIReps{epsidx}.time(begtime:endtime), ...
    AIReps{epsidx}.f(begtime:endtime),"-.",'linewidth',2)

  begtime = find(EPIReps{epsidx}.time>stimeeps ,1);
  endtime = find(EPIReps{epsidx}.time>stimeeps+timelen ,1);
  plot(EPIReps{epsidx}.time(begtime:endtime), ...
    EPIReps{epsidx}.f(begtime:endtime),"-r",'linewidth',2)
end
axis([0 1.05 200 700])
legend("PIR-\epsilon  = 10^{-2}","AIR-\epsilon_{0} = 10^{-2}", ...
  "EPIR-\epsilon_{0}=10^{-2}","PIR-\epsilon  = 10^{-4}", ...
  "AIR-\epsilon_{0} = 10^{-4}","EPIR-\epsilon_{0}=10^{-4}")       
xlabel("CPU-time(s)"); ylabel("Objective F(x)")
% 
% begtime = find(AIReps{1}.time>stimeeps ,1);
% endtime = find(AIReps{1}.time>stimeeps+timelen ,1);
% plot(AIReps{1}.time(begtime:endtime),AIReps{1}.f(begtime:endtime),"-.ko")
% 
% begtime = find(EPIReps{1}.time>stimeeps ,1);
% endtime = find(EPIReps{1}.time>stimeeps+timelen ,1);
% plot(EPIReps{1}.time(begtime:endtime),EPIReps{1}.f(begtime:endtime),"-.bs")
hold off
% legend("PIR-\epsilon = 1","PIR-\epsilon  = 10^{-1}","PIR-\epsilon  = 10^{-2}",...
% "PIR-\epsilon = 10^{-3}","PIR-\epsilon = 10^{-4}",...
% "AIR-\epsilon_{0}=1","EPIR-\epsilon_{0}=1")
    %% polt sensitive of eps for ds_ProxIRNN and robust for AIRNN/EIRNRI
% iteration VS f
figure(1)
  pbeg = 10;
  pend = 1e2;
  % for epsidx =1:length(orieps_spl)
  for epsidx =2:3
    plot(PIReps{epsidx}.f(pbeg:pend),"^"); hold on
    plot(AIReps{epsidx}.f(pbeg:pend),"-.+")
    plot(EPIReps{epsidx}.f(pbeg:pend),"-.s")
  end
legend("PIR-\epsilon = 1","PIR-\epsilon  = 10^{-1}",...
  "PIR-\epsilon  = 10^{-2}","PIR-\epsilon = 10^{-3}",...
  "PIR-\epsilon = 10^{-4}","AIR-\epsilon_{0}=1","EPIR-\epsilon_{0}=1")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ?????????????????????????????????????????????????????????????
%% ------------------ sensitive of eps and plot
rcSen = 15; X0 = (1+randn(nr,rcSen))*(randn(rcSen,nc)); 
optionsP.Rel = Y; 
optionsP.eps = 5e-1;
optionsP.max_iter = itmax; 
PIR = ds_ProxIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
optionsP.mu = 0.8; 
AIR = ds_AdaIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
optionsP.alpha = 0.7; 
EPIR = ds_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 

%%
  plot(PIRx,PIR.f(1:pPIR),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',1);
  plot(EPIRx,EPIR.f(1:pEPI),'-.g','linewidth',1);
%   title("Objective"); 
  xlabel("iteration"); ylabel("F(x)")
  legend("PIRNN","AIRNN","EIRNRI")
    
    
    %%
  % % ------------ obj plot 
  figure(3)
  plot(PIRx,PIR.f(1:pPIR),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',1);
  plot(EPIRx,EPIR.f(1:pEPI),'-.g','linewidth',1);hold off
%   title("Objective"); 
  xlabel("iteration"); ylabel("F(x)")
  legend("PIRNN","AIRNN","EIRNRI")

  % ------------ rank plot
  figure(4)
  plot(PIRx,PIR.rank(1:pPIR),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.rank(1:pAIR),'--b','linewidth',1);
  plot(EPIRx,EPIR.rank(1:pEPI),'-.g','linewidth',1);hold off 
%   title("rank of iterations"); 
  xlabel("iteration"); ylabel("rank")
  legend("PIRNN","AIRNN","EIRNRI")

  
%% ----------------------------------------------------- end  
  % ------------  time plot
%%
optionsP.alpha = 1e-1; 
EPIR = ds_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
%% plot the Relative error and Objective 
PIRx = (1:1:PIR.iterTol); 
AIRx = (1:1:AIR.iterTol); 
EPIRx = (1:1:EPIR.iterTol); 
% ------------ RelErr plot 
figure(1)
plot(PIRx,log10(PIR.RelErr),'.r','linewidth',1);hold on
plot(AIRx,log10(AIR.RelErr),'--b','linewidth',1);
plot(EPIRx,log10(EPIR.RelErr),'-.g','linewidth',1);hold off
title("RelErr "); 
xlabel("iteration"); ylabel("RelErr")
legend("PIRNN","AIRNN","EIRNRI")

% ------------ RelDist plot 
figure(2)
plot(PIRx,log10(PIR.RelDist),'.r','linewidth',1);hold on
plot(AIRx,log10(AIR.RelDist),'--b','linewidth',1);
plot(EPIRx,log10(EPIR.RelDist),'-.g','linewidth',1);hold off
title("RelDist "); 
xlabel("iteration"); ylabel("RelDist")
legend("PIRNN","AIRNN","EIRNRI")
% % ------------ obj plot 
figure(3)
plot(PIRx,PIR.f,'.r');hold on; plot(AIRx,AIR.f,'--b');
plot(EPIRx,EPIR.f,'-.g');hold off
title("Objective"); 
xlabel("iteration"); ylabel("F(x)")
legend("PIRNN","AIRNN","EIRNRI")
% % plot(px,Fp,'-+r','linewidth',1);hold on
% % plot(px,Fa,'--sb','linewidth',1);hold on
% % plot(px,Fe,'-.g','linewidth',1);hold on
% % legend("PIRNN","AIRNN","EIRNRI")
% plot(px,Fp,'-r');hold on; plot(px,Fa,'--b');
% plot(px,Fe,'-.g');hold off
% legend("PIRNN","AIRNN","EIRNRI")

% figure(3)

% ------------ rank plot
figure(4)
plot(PIRx,PIR.rank,'.r');hold on; plot(AIRx,AIR.rank,'--b');
plot(EPIRx,EPIR.rank,'-.g');hold off 
title("rank of iterations")
xlabel("iteration"); ylabel("rank")
legend("PIRNN","AIRNN","EIRNRI")
% ------------  time plot
figure(5)
%% Time 
AIRtime = []; EPIR = []; EPIRtime = [] ;
%%
orieps = 1e-8;optionsP.eps = orieps; 
%%

optionsP.max_iter = itmax;
optionsP.Rel = Y; 
optionsP.eps = orieps; 
optionsP.mu = 0.3;

mu = 1.1:0.45:2; 

stay = zeros(length(mu),1) ;
for stm =1:length(mu)
  optionsP.mu = mu(stm) ;
  upalpha = sqrt(mu(stm)/(mu(stm)+2)); 
  adalpha = 5e-2:0.05:upalpha ;
  stay(stm) = length(adalpha); 
  AIR = ds_AdaIRNN(X0,Y,sp, lambda, mask, tol, optionsP);
  AIRtime = [AIRtime,AIR]; 
  optionsP.mu = mu(stm);
  for i = 1:length(adalpha)
    optionsP.alpha = adalpha(i);  
    sol = ds_EPIRNN(X0,Y,sp, lambda, mask, tol, optionsP); 
    EPIR = [EPIR,sol];
  end
%   EPIRtime = [EPIRtime;EPIR];
end
%%
EPIRtime - AIR.time


%%
while RelDist_PIRNN>tol && iter <= IterTol  
  iter = iter + 1 ; 
  Rk = rank(X0) ; 
  [ud,sigma,vd] = svd(X0);
  RelDist_PIRNN = norm(ud(1:Rk,:)*Gradf(X0)*vd(1:Rk,:)'+...
    lambda*sp*spdiags(diag(sigma).^(sp-1),0,Rk,Rk),'fro')/norm(X0,'fro'); 
%   rew = sp*(sigma+Reps).^(sp-1) ;
  [U,S,V] = svd(X0 - Gradf(X0)/mu) ;
  NewS = diag(S) - sp*(diag(sigma)+Reps*ones(rc,1)).^(sp-1)*2/mu ;
  X1 = U*spdiags(NewS.*(NewS>0),0,m,n)*V' ; 

  if (iter == 1) || (mod(iter, 50) == 0) || (RelDist_PIRNN < tol)
    fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
            iter, RelDist_PIRNN, rank(X0), Objf(X0)); 
                  % nnz(X1(~unobserved)),
  end
  X0 = X1 ; 
end
X_PIRNN = X0;
% -------------------------------- AIRNN --------------------------------
%
EpsScalar = 0.7 ; 
RelDist = 1; tol = 1e-5; iter = 0; IterTol = 500;  
X0 = OriX ; 
while RelDist>tol && iter <= IterTol  
  [ud,sigma,vd] = svd(X0);
  Aireps = Aireps.*(diag(sigma)>1e-13)*EpsScalar +...
    Aireps.*(diag(sigma)<=1e-13) ;
  iter = iter + 1 ; 
  Rk = rank(X0) ; 
  
  RelDist = norm(ud(1:Rk,:)*Gradf(X0)*vd(1:Rk,:)'+...
    lambda*sp*spdiags(diag(sigma).^(sp-1),0,Rk,Rk),'fro')/norm(X0,'fro'); 
%   rew = sp*(sigma+Reps).^(sp-1) ;
  [U,S,V] = svd(X0 - Gradf(X0)/mu) ;
  NewS = diag(S) - sp*(diag(sigma)+Aireps).^(sp-1)*2/mu ;
  X1 = U*spdiags(NewS.*(NewS>0),0,m,n)*V' ; 

  if (iter == 1) || (mod(iter, 50) == 0) || (RelDist < tol)
            fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
                    iter, RelDist, rank(X1),Objf(X0) );
                  %nnz(X1(~unobserved))
  end
  X0 = X1 ; 
end
X_AIRNN = X0 ; 
% \end

%% Nuclear norm
r=2*rt;
par=[0.02 0.03 0.04];
for i=1:length(par)
  for j=1:3
  tic;[Xr{1,i}(:,:,j)]=inexact_alm_rpca(Xn(:,:,j),par(i),1e-8,1000);
  t_temp_n(i)=toc;
  tic;
  [Xr{1,i}(:,:,j),E_r] = RobustPCA(Xn(:,:,j), par(i), par(i)*10, 1e-5, 500);
  t_temp_n(i)=toc;
  end
  e{1,i}=norm(Xr{1,i}(:)-X(:))/norm(X(:));
end
%% F-Nuclear norm
par=[0.02 0.03 0.04];
opt.d=r;
opt.u=1;
for i=1:length(par)
    for j=1:3
        [Xr{2,i}(:,:,j)]=RPCA_FNuclear_ADMM(Xn(:,:,j),par(i),opt);
    end
    e{2,i}=norm(Xr{2,i}(:)-X(:))/norm(X(:));
end
%% FGSR 2/3
par=[0.02 0.03 0.04];
opt.regul_B='L2';
opt.d=r;
opt.u=1;opt.maxiter=1500;
for i=1:length(par)
    for j=1:3
        [Xr{3,i}(:,:,j),E{3,i},output]=RPCA_FGSR_ADMM(Xn(:,:,j),par(i),opt);
    end
    e{3,i}=norm(Xr{3,i}(:)-X(:))/norm(X(:));
end
%%%% FGSR 1/2
% par=[0.01 0.02 0.03];
% opt.regul_B='L21';
% opt.d=r;
% opt.u=0.1;opt.maxiter=1500;
% for i=1:length(par)
%     for j=1:3
%         [Xr{4,i}(:,:,j),E{4,i},output]=RPCA_FGSR_ADMM(Xn(:,:,j),par(i),opt);
%     end
%     e{4,i}=norm(Xr{4,i}(:)-X(:))/norm(X(:));
% end

%%
figure;
NC=5;
dp=[-0.05 0 0.035 0];
h=subplot(2,NC,1);imshow(X);ht=title({'Clean image';'ground truth'});
set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,2);imshow(Xn);ht=title({'Corrupted image';'40% salt&pepper noise'}');
set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,3);imshow(Xr{1});ht=title({'RPCA(nuclear norm)';'relative recovery error=0.033'});
set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,4);imshow(Xr{2});ht=title({'RPCA(F-nuclear norm)';'relative recovery error=0.022'});
set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,5);imshow(Xr{3});title({'RPCA(FGSR-2/3)';'relative recovery error=0.005'});
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);



