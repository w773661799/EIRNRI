%% test the AIRNN / EPIRNN with random data and different mu
clc,clear,format long 
rng(22)
nr = 150; nc = 150; r = 15 ;
% --------------- Synthetic data ---------------
xb = 1+randn(nr,r); xc = 1+randn(r,nc) ;
Y = xb*xc ; 

% --------------- random mask ---------------
M_org = zeros(nr,nc); 
missrate = 0.8; 
for i=1:nc  
    idx = 1:1:nr;
    randidx=randperm(nr,nr); % 随机[n] 中的 k 个 index
    M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~M_org; 
X0 = 10+randn(nr,nc); Xm=Y.*mask; 

% --------------- parameters ---------------
orieps = 10 ;
sp = 0.4 ; % sp norm 
lambda = 1e-4*norm(Y,inf);
tol = 5e-5; 
itmax = 1e5; 
%  
AIRtime = {}; EPIRtime = {} ;
%%
optionsP.max_iter = itmax;
optionsP.Rel = Y; 
optionsP.eps = orieps; 
optionsP.Scalar = 0.9;

mu = 1.1:0.2:2.1 ;

stay = zeros(length(mu),1) ;
for stm =1:length(mu)
  optionsP.mu = mu(stm) ;
%   upalpha = sqrt(mu(stm)/(mu(stm)+2)); 
%   adalpha = 5e-2:0.1:upalpha ;
  adalpha = 5e-2:0.05:0.95; 
  stay(stm) = length(adalpha); 
  AIR_sol = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
  AIRtime{stm} = AIR_sol; 
  optionsP.mu = mu(stm);
  EPIR = {};
  for i = 1:length(adalpha)
    optionsP.alpha = adalpha(length(adalpha)-i+1);  
    EPIR_sol = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
    EPIR{i} = EPIR_sol;
  end
  EPIRtime{stm} = EPIR;
end

%% plot the error and Objf
AIR_plot = AIR_sol ; 
EPIR_plot = EPIR_sol ; 
AIR_plotx = (1:1:AIR_plot.iterTol)/100; 
EPIR_plotx = (1:1:EPIR_plot.iterTol)/100; 
% % ------------ RelErr plot 
figure(1)
plot(AIR_plotx,log10(AIR_plot.RelErr),'--b','linewidth',1);hold on
plot(EPIR_plotx,log10(EPIR_plot.RelErr),'-.g','linewidth',1);hold off
title("RelErr "); 
xlabel("iteration"); ylabel("RelErr")
legend("AIRNN","EPIRNN")

% % ------------ RelDist plot 
figure(2)
plot(AIR_plotx,log10(AIR_plot.RelDist),'--b','linewidth',1); hold on
plot(EPIR_plotx,log10(EPIR_plot.RelDist),'-.g','linewidth',1);hold off
title("RelDist "); 
xlabel("iteration"); ylabel("RelDist")
legend("AIRNN","EPIRNN")
% % ------------ obj plot 
figure(3)
plot(AIR_plotx,AIR_plot.f,':k');hold on; 
plot(EPIR_plotx,EPIR_plot.f,'-.g');hold off
title("Objective"); 
xlabel("iteration"); ylabel("F(x)")
legend("AIRNN","EPIRNN")

% % ------------ rank plot
figure(4)
plot(AIR_plotx,AIR_plot.rank,'--b');hold on;
plot(EPIR_plotx,EPIR_plot.rank,'-.g');hold off 
title("rank of iterations")
xlabel("iteration"); ylabel("rank")
legend("AIRNN","EPIRNN")
% ------------  time plot
figure(5)

%%
