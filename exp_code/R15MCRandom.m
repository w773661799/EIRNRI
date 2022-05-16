%% test the PIRNN / AIRNN / EPIRNN with random data  
% this experiment shows the EPIRNN will convergence faster than PIRNN
% the AIRNN has the similar convergence rate with PIRNN
clc,clear,format long 
rng(22)
nr = 150; nc = 150; r = 15 ;
% --------------- Synthetic data ---------------
xb = randn(nr,r); xc = randn(r,nc) ;
Y = xb*xc ; 
% --------------- random mask ---------------
M_org = zeros(nr,nc); 
missrate = 0.5; 
for i=1:nc  
    idx = 1:1:nr;
    randidx=randperm(nr,nr); % random sequence
    M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~M_org; Xm=Y.*mask; 
% --------------- parameters ---------------
lambda = 1e-4*norm(Xm,inf);
itmax = 2e4; 
sp = 0.1; 
tol = 1e-5; 
%% basic algorithm for sp=0.5 ,SR=0.5
  % Initial point
  % with default eps = eps(1)
  rcSen = 15; X0 = (1+randn(nr,rcSen))*(randn(rcSen,nc)); 
  optionsP.Rel = Y; 
  optionsP.max_iter = itmax; 
  optionsP.eps = eps(1);
  PIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
  optionsP.Scalar = 0.8; 
  AIR = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
  optionsP.alpha = 5e-1; 
  EPIR = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
    %% plot 
    pPIR = min(itmax,PIR.iterTol); pAIR = min(itmax,AIR.iterTol);
    pEPI = min(itmax,EPIR.iterTol); 
    PIRx = (1:1:pPIR); AIRx = (1:1:pAIR); EPIRx = (1:1:pEPI); 
    % ------------ RelErr plot 
    h = figure(1);
%     set (gca,'position',[0.1,0.1,0.9,0.9] );
    set(h,'Position',[500 500 1500 500]);
%     subplot(1,3,1)
subplot('Position',[0.05,0.1,0.28,0.85])
    plot(PIRx,log10(PIR.RelErr(1:pPIR)),':ko','linewidth',1);hold on
    plot(AIRx,log10(AIR.RelErr(1:pAIR)),'--b^','linewidth',1);
    plot(EPIRx,log10(EPIR.RelErr(1:pEPI)),'-.rs','linewidth',1); hold off
    xlabel("iteration"); ylabel("log(RelErr)")
    legend("PIRNN","AIRNN","EPIRNN")
subplot('Position',[0.38,0.1,0.28,0.85])
%     subplot(1,3,2,'position',[0.35,0,0.3,1])
    plot(PIRx,log10(PIR.RelDist(1:pPIR)),':ko','linewidth',1);hold on
    plot(AIRx,log10(AIR.RelDist(1:pAIR)),'--b^','linewidth',1);
    plot(EPIRx,log10(EPIR.RelDist(1:pEPI)),'-.rs','linewidth',1);hold off
    xlabel("iteration"); ylabel("log(RelDist)")
    legend("PIRNN","AIRNN","EPIRNN")
subplot('Position',[0.71,0.1,0.28,0.85])
%     subplot(1,3,3,'position',[0.7,0,0.3,1])
    plot(PIRx,PIR.f(1:pPIR),':ko','linewidth',1);hold on; 
    plot(AIRx,AIR.f(1:pAIR),'--b^','linewidth',1);
    plot(EPIRx,EPIR.f(1:pEPI),'-.rs','linewidth',1);hold off
    xlabel("iteration"); ylabel("F(x)")
    legend("PIRNN","AIRNN","EPIRNN")
    %%
%     plot(PIRx,log10(PIR.RelErr(1:pPIR)),':.k','linewidth',1);
%     plot(AIRx,log10(AIR.RelErr(1:pAIR)),'--b','linewidth',1);
%     plot(EPIRx,log10(EPIR.RelErr(1:pEPI)),'-.r','linewidth',1);
%     xlabel("iteration"); ylabel("log(RelErr)")
%     legend("PIRNN","AIRNN","EPIRNN")
%     % ------------ RelDist plot 
%     figure(2)
%     plot(PIRx,log10(PIR.RelDist(1:pPIR)),':.k','linewidth',1);hold on
%     plot(AIRx,log10(AIR.RelDist(1:pAIR)),'--b','linewidth',1);
%     plot(EPIRx,log10(EPIR.RelDist(1:pEPI)),'-.r','linewidth',1);hold off
%     xlabel("iteration"); ylabel("log(RelDist)")
%     legend("PIRNN","AIRNN","EPIRNN")
%     % ------------ objective value plot 
%     figure(3)
%     plot(PIRx,PIR.f(1:pPIR),':.k','linewidth',1);hold on; 
%     plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',1);
%     plot(EPIRx,EPIR.f(1:pEPI),'-.r','linewidth',1);hold off
%     xlabel("iteration"); ylabel("F(x)")
%     legend("PIRNN","AIRNN","EPIRNN")
% -/end -----------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ------------------ sensitive of alpha and plot
Lalpha = [0 0.1 0.3 0.5 0.7 0.9];
for i = 1:length(Lalpha)
  optionsP.alpha = Lalpha(i) ;
  LEPIR{i} = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
end
  %% plot 
  figure(1) % RelErr
    pltLEPx = min(itmax,LEPIR{i}.iterTol);
    plot(log10(LEPIR{1}.RelErr),'-.r','linewidth',1);hold on
    plot(log10(LEPIR{2}.RelErr),':or','linewidth',1);
    plot(log10(LEPIR{3}.RelErr),':+k','linewidth',1);
    plot(log10(LEPIR{4}.RelErr),':.k','linewidth',1);
    plot(log10(LEPIR{5}.RelErr),'-b','linewidth',1);
    plot(log10(LEPIR{6}.RelErr),'--.b','linewidth',1);hold off
    xlabel("iteration"); ylabel("log(RelErr)")
    legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
    "\alpha = 0.7","\alpha = 0.9")
  figure(2) % Errdist
    plot(log10(LEPIR{1}.RelDist),'-.r','linewidth',1);hold on
    plot(log10(LEPIR{2}.RelDist),':or','linewidth',1);
    plot(log10(LEPIR{3}.RelDist),':+k','linewidth',1);
    plot(log10(LEPIR{4}.RelDist),':.k','linewidth',1);
    plot(log10(LEPIR{5}.RelDist),'-b','linewidth',1);
    plot(log10(LEPIR{6}.RelDist),'--.b','linewidth',1);hold off
    xlabel("iteration"); ylabel("log(RelDist)")
    legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
    "\alpha = 0.7","\alpha = 0.9")
  figure(3) % obj
    plot(log10(LEPIR{1}.f),'-.r','linewidth',1);hold on
    plot(log10(LEPIR{2}.f),':or','linewidth',1);
    plot(log10(LEPIR{3}.f),':+k','linewidth',1);
    plot(log10(LEPIR{4}.f),':.k','linewidth',1);
    plot(log10(LEPIR{5}.f),'-b','linewidth',1);
    plot(log10(LEPIR{6}.f),'--.b','linewidth',1);hold off
    xlabel("iteration"); ylabel("F(X)")
    legend("\alpha = 0","\alpha = 0.1","\alpha = 0.3","\alpha = 0.5",...
    "\alpha = 0.7","\alpha = 0.9")
% -/end -----------------------------------------------------------------

%% ?????????????????????????????????????????????????????????????
%% ------------------ sensitive of eps and plot
rcSen = 15; X0 = (1+randn(nr,rcSen))*(randn(rcSen,nc)); 
optionsP.Rel = Y; 
optionsP.eps = 5e-1;
optionsP.max_iter = itmax; 
PIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
optionsP.Scalar = 0.8; 
AIR = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
optionsP.alpha = 0.7; 
EPIR = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 

%%
  plot(PIRx,PIR.f(1:pPIR),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',1);
  plot(EPIRx,EPIR.f(1:pEPI),'-.g','linewidth',1);
%   title("Objective"); 
  xlabel("iteration"); ylabel("F(x)")
  legend("PIRNN","AIRNN","EPIRNN")
    
    
    %%
  % % ------------ obj plot 
  figure(3)
  plot(PIRx,PIR.f(1:pPIR),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.f(1:pAIR),'--b','linewidth',1);
  plot(EPIRx,EPIR.f(1:pEPI),'-.g','linewidth',1);hold off
%   title("Objective"); 
  xlabel("iteration"); ylabel("F(x)")
  legend("PIRNN","AIRNN","EPIRNN")

  % ------------ rank plot
  figure(4)
  plot(PIRx,PIR.rank(1:pPIR),':.r','linewidth',1);hold on; 
  plot(AIRx,AIR.rank(1:pAIR),'--b','linewidth',1);
  plot(EPIRx,EPIR.rank(1:pEPI),'-.g','linewidth',1);hold off 
%   title("rank of iterations"); 
  xlabel("iteration"); ylabel("rank")
  legend("PIRNN","AIRNN","EPIRNN")

  
%% ----------------------------------------------------- end  
  % ------------  time plot
%%
optionsP.alpha = 1e-1; 
EPIR = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
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
legend("PIRNN","AIRNN","EPIRNN")

% ------------ RelDist plot 
figure(2)
plot(PIRx,log10(PIR.RelDist),'.r','linewidth',1);hold on
plot(AIRx,log10(AIR.RelDist),'--b','linewidth',1);
plot(EPIRx,log10(EPIR.RelDist),'-.g','linewidth',1);hold off
title("RelDist "); 
xlabel("iteration"); ylabel("RelDist")
legend("PIRNN","AIRNN","EPIRNN")
% % ------------ obj plot 
figure(3)
plot(PIRx,PIR.f,'.r');hold on; plot(AIRx,AIR.f,'--b');
plot(EPIRx,EPIR.f,'-.g');hold off
title("Objective"); 
xlabel("iteration"); ylabel("F(x)")
legend("PIRNN","AIRNN","EPIRNN")
% % plot(px,Fp,'-+r','linewidth',1);hold on
% % plot(px,Fa,'--sb','linewidth',1);hold on
% % plot(px,Fe,'-.g','linewidth',1);hold on
% % legend("PIRNN","AIRNN","EPIRNN")
% plot(px,Fp,'-r');hold on; plot(px,Fa,'--b');
% plot(px,Fe,'-.g');hold off
% legend("PIRNN","AIRNN","EPIRNN")

% figure(3)

% ------------ rank plot
figure(4)
plot(PIRx,PIR.rank,'.r');hold on; plot(AIRx,AIR.rank,'--b');
plot(EPIRx,EPIR.rank,'-.g');hold off 
title("rank of iterations")
xlabel("iteration"); ylabel("rank")
legend("PIRNN","AIRNN","EPIRNN")
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
optionsP.Scalar = 0.3;

mu = 1.1:0.45:2; 

stay = zeros(length(mu),1) ;
for stm =1:length(mu)
  optionsP.mu = mu(stm) ;
  upalpha = sqrt(mu(stm)/(mu(stm)+2)); 
  adalpha = 5e-2:0.05:upalpha ;
  stay(stm) = length(adalpha); 
  AIR = MC_AIRNN(X0,Y,sp, lambda, mask, tol, optionsP);
  AIRtime = [AIRtime,AIR]; 
  optionsP.mu = mu(stm);
  for i = 1:length(adalpha)
    optionsP.alpha = adalpha(i);  
    sol = MC_EPIRNN(X0,Y,sp, lambda, mask, tol, optionsP); 
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
%     tic;[Xr{1,i}(:,:,j)]=inexact_alm_rpca(Xn(:,:,j),par(i),1e-8,1000);t_temp_n(i)=toc;
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
h=subplot(2,NC,1);imshow(X);ht=title({'Clean image';'ground truth'});set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,2);imshow(Xn);ht=title({'Corrupted image';'40% salt&pepper noise'}');set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,3);imshow(Xr{1});ht=title({'RPCA(nuclear norm)';'relative recovery error=0.033'});set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,4);imshow(Xr{2});ht=title({'RPCA(F-nuclear norm)';'relative recovery error=0.022'});set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,5);imshow(Xr{3});title({'RPCA(FGSR-2/3)';'relative recovery error=0.005'});
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);



