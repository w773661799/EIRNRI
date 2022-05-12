%% AIRNN Test Nuclear norm
clc,clear
rng(22);format long 
X_img=imread('sistlibrary.jpeg');
% X=imread('im2.jpg');
% X=imresize(X,0.5);% im1 0.5
X=double(X_img)/255;
% [U,S,V]=svd(X);
% figure(1), imshow(X)

% rt = ceil(size(X(:,:,1),2)/3);
rt = ceil(min(size(X(:,:,1)))/5); 
% SVD : U*S*V' = svd(X)
%       VecSigma = svd(X)
for i=1:3
  [U,S,V]=svd(X(:,:,i));
  Sigma(:,:,i) = S ; 
  Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
end

%% ------------------------ random mask ------------------------
[nr,nc] = size(X(:,:,1));
missrate = 0.5;
mask = zeros(nr,nc);
for i=1:nc  
    idx = 1:1:nr;
    randidx=randperm(nr,nr); % 随机[n] 中的 k 个 index
    mask(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~mask;
X0 = (1+randn(nr,rt))*(randn(rt,nc));
%% ------------------------ parameters ------------------------
% 初始矩阵 + 线性参数 + 正则参数
XM = mask.*Xt;
sp = 0.5;
tol = 1e-5 ;
options.max_iter = 1e4; 
options.eps = 1e-8; 
options.mu = 1.1;
options.KLopt = 1e-5; 
% options.KLopt = 1e-6 ;  

% PIRNN
optionsP = options; 
% AIRNN
optionsA = options; optionsA.Scalar = 0.1; optionsA.eps = 1;
% EPIRNN
optionsEP = optionsA; optionsEP.alpha = 0.8;
Parsol = {}; 
for i =1:3
  Xm = XM(:,:,i);
  lambda = 5e-3*norm(Xm,"fro");
  PIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
  AIR = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsA);
  EPIR = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsEP);
  XPIR(:,:,i) = PIR.Xsol; Parsol{i,1} = PIR;
  XAIR(:,:,i) = AIR.Xsol; Parsol{i,2} = AIR;
  XEPIR(:,:,i) = EPIR.Xsol; Parsol{i,3} = EPIR;
end
%% figure and PSNR
% PSNR 
  PSNR = @(x,y,z)(10*log10( (255^2*3*nr*nc)/(norm(Xt(:,:,1)-x,"fro")+norm(Xt(:,:,2)-y,"fro")+...
    norm(Xt(:,:,1)-z,"fro")) )) ;

  PSNRXsol = [PSNR(XPIR(:,:,1),XPIR(:,:,2),XPIR(:,:,3));
    PSNR(XAIR(:,:,1),XAIR(:,:,2),XAIR(:,:,3));
    PSNR(XEPIR(:,:,1),XEPIR(:,:,2),XEPIR(:,:,3))
    ] 
% iterations
  Iter = [ceil((Parsol{1,1}.iterTol+Parsol{2,1}.iterTol +Parsol{3,1}.iterTol)/3);
    ceil((Parsol{1,2}.iterTol+Parsol{2,2}.iterTol +Parsol{3,2}.iterTol)/3);
    ceil((Parsol{3,1}.iterTol+Parsol{2,3}.iterTol +Parsol{3,3}.iterTol)/3);
  ]

% rank with iterations
  figure(1)
  for i = 1:3
    set(gca,'LooseInset',get(gca,'TightInset'))
    plot(Parsol{3*i-2}.rank,'linewidth',2);hold on
  end
  xlabel("iteration"); ylabel("rank")
  legend("PIRNN","AIRNN","EPIRNN")
  
  figure(2)
  for i = 1:3
    set(gca,'LooseInset',get(gca,'TightInset'))
    plot(Parsol{3*i-1}.rank,'linewidth',2);hold on
  end
  xlabel("iteration"); ylabel("rank")
  legend("PIRNN","AIRNN","EPIRNN")
  
  figure(3)
  for i = 1:3
    set(gca,'LooseInset',get(gca,'TightInset'))
    plot(Parsol{3*i}.rank,'linewidth',2);hold on
  end
  xlabel("iteration"); ylabel("rank")
  legend("PIRNN","AIRNN","EPIRNN")
%%
% Real image and restored image
%   
figure("units","normalized","position",[0, 0, 0.4, 0.33])
  imshow(Xt,"border","tight","initialmagnification","fit"); 
figure("units","normalized","position",[0, 0, 0.4, 0.33])  
  imshow(XM,"border","tight","initialmagnification","fit");
figure("units","normalized","position",[0, 0, 0.4, 0.33])  
  imshow(XPIR,"border","tight","initialmagnification","fit");
figure("units","normalized","position",[0, 0, 0.4, 0.33])  
  imshow(XAIR,"border","tight","initialmagnification","fit");
figure("units","normalized","position",[0, 0, 0.4, 0.33])  
  imshow(XEPIR,"border","tight","initialmagnification","fit"); 
%%  
  subplot('Position',[0 0.8 0.2 0.2]); imshow(Xt); 
  hold on; xlabel("(a)")
  subplot('Position',[0.2 0.8  0.2 0.2]); imshow(XM)
  xlabel("(b)")
  subplot('Position',[0.4 0.8 0.2 0.2]); imshow(XPIR)
  xlabel("(c)")
  subplot('Position',[0.6 0.8 0.2 0.2]); imshow(XAIR)
  xlabel("(d)")
  subplot('Position',[0.8 0.8 0.2 0.2]); imshow(XEPIR);
  xlabel("(e)");  hold off
%%  

  
  
  
  
%% ??? 
%   subplot(3,2,1); imshow(X_img)
%   subplot(5,1,1); imshow(Xt)
%   subplot(5,1,2); imshow(XM)
%   subplot(5,1,3); imshow(XPIR)
%   subplot(5,1,4); imshow(XAIR)
%   subplot(5,1,5); imshow(XEPIR)
subplot("Position",[left bottom width height])
Pos1 = [0.13 0.5 0.13 0.5];
  sub1 = subplot(1,5,1); set(sub1,"Position",Pos1)
  imshow(Xt)
  subplot(1,5,2); imshow(XM)
  subplot(1,5,3); imshow(XPIR)
  subplot(1,5,4); imshow(XAIR)
  subplot(1,5,5); imshow(XEPIR)
% %% test R / G / B recovery  
% % test to find the optimal eps for PIRNN and then accelerate with opt_alpha   
% lseps =[]; 
% for i =5:10
%   for j=5:-4:1
%     lseps = [lseps,j*10^(-i)]; 
%   end
% end

% XPIR = {}; 
% XAIR = {};
% XEPIR = {};
% for i = 1:length(lseps)
%   optionsP.eps = lseps(i)
%   PIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
%   XPIR{i} = PIR; 
% end

  % %  choise a weps for PIRNN 
  
  % optionsP.eps = 1e-5

% % opt_eps ~~ 10-6 ... 1e-10
  %% find the optimal scalar and alpha for accelerate 
  
%   optionsEP.eps = 1e-5;
  
  lsalpha = 0.2:0.2:0.8;
  i=4
%   for i=1:length(lsalpha)
  optionsEP.alpha = lsalpha(i)
  
 
%   end
%   optionsEP.Scalar = lsScalar(i); 
%   optionsEP.alpha = lsAlpha(j); 
  
  
  %%
  lsScalar = 0.3:0.6:0.9;
  lsAlpha = 0.3:0.1:0.8;
  lsEPIR = {};
  for i  = 1:length(lsScalar)
    optionsEP.Scalar = lsScalar(i); 
    for j =1:length(lsAlpha)
      optionsEP.alpha = lsAlpha(j)
      EPIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsEP)
      lsEPIR{i,j} = EPIR; 
    end
  end
  %%
i=1; j=1; 
Xm = XM(:,:,i);
XPIR = {}; XEPIR = {}; 
% % for i = 1:10
optionsP.mu = 1+1e-1;
optionsP.eps = 1e-6;
% %   for j = 1:9
%     lambda = 1*1e-2*norm(Xm(:,:,1),inf);
    PIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
    XPIR{i,j} = PIR.Xsol; 
% %     AIR = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
% %     XAIR(:,:,i) = AIR.Xsol;
    EPIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
    XEPIR{i,j} = EPIR.Xsol; 
%     fprintf("the parameters, mu:%d\t lambda:%d\n",i,j)
%   end
% end
% mu \ lambda = 1.1 \ {2,3,4}e-2;  
% 1.2 \ {3,4}e-2 
% 1.3 \ {3}e-2 
% 1.5 \ {3,4}e-2
% 

    %% recovery 
    for i = 1:3
      Xm = XM(:,:,i); 
      lambda = 1e-2*norm(X(:,:,1),inf);
      PIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
      XPIR(:,:,i) = PIR.Xsol ; 
      AIR = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
      XAIR(:,:,i) = AIR.Xsol;
      EPIR = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP);
      XEPIR(:,:,i) = EPIR.Xsol; 
    end
%%
for i =1:3
XP(:,:,i) = XPIR(:,:,i).Xsol;
XA(:,:,i) = XAIR(:,:,i).Xsol; 
XE(:,:,i) = XEPIR(:,:,i).Xsol;
end

%%
[m,n] = size(S) ; rc = min(m,n) ;
VecSigma(:,i) = diag(Sigma(:,:,i)) ;  
Reps = 1 ; % perturbation

% \初始化
Y = X(:,:,1) ; 
lambda = 1e-5*norm(Y,inf) ;
mu = 1.1 ;
X0 = 10+10*randn(m,n) ; OriX = X0 ; 
RelDist_PIRNN = 1;
tol = 1e-5 ;
Gradf = @(x)(x-Y) ; 
Objf = @(x)(norm(x-Y,'fro')/2+lambda*norm(svd(x),sp))^(sp) ; 
iter = 0 ; 
% ---
unobserved = isnan(X0);
X0(unobserved) = 0;
IterTol = 500 ; 
Aireps = Reps*ones(rc,1) ; 
%% test the PIRNN / AIRNN / EPIRNN 
M_org = X0<5e-1 ; [nr,nc] = size(X0) ; 
missrate = 0.8 
M_t=ones(nr,nc);
% random mask
for i=1:nc 
    idx=find(M_org(:,i)==1); %找到 第一次 mask 的 logical parameter 
    lidx=length(idx); 
    temp=randperm(lidx,ceil(lidx*missrate)); % 随机[n] 中的 k 个 index   
    temp=idx(temp);
    M_t(temp,i)=0; 
end
%
M=M_t.*M_org;
Xm=Y.*M;
vd=[20];
%% 
clc
format long 
% X0 = Xm ; 

mask = ~M_org ; 
r = 15 ;
xb = 1+randn(m,r); xc = 2+randn(r,n) ;
Y = xb*xc ; 

X0 = 10+randn(m,n) ; 

tol = 1e-6 ;
%% test for Synthetic data
clc
m = 150; n = 150; r = 15; 
mask = unifrnd(0,1,m,n)>1e-2; 
Brnd = randn(m,r)+1 ; Crnd = rand(r,n) ; 
Y = Brnd*Crnd.*mask ;
X0 = rand(m,n) ;  TeX = X0 ;

%% 
format long 
X0 = TeX ; 
%%
sp = 0.4 ; % sp norm 
lambda = 1e-4*norm(Y,inf);
tol = 5e-5 ;
itmax = 5e3; 
optionsP.max_iter = itmax;
optionsP.Rel = Brnd*Crnd; 
optionsP.eps = 1 ; 
optionsP.Scalar = 0.9;
PIR = MC_PIRNN(X0,Y,sp, lambda, mask, tol, optionsP);
AIR = MC_AIRNN(X0,Y,sp, lambda, mask, tol, optionsP);

options.Rel = Brnd*Crnd; 
options.alpha = 4e-1; 
options.eps = 1;   
options.max_iter = itmax ;
options.Scalar = 0.9;
EPIR = MC_EPIRNN(X0,Y,sp, lambda, mask, tol, options); 
%% Tiem 

mu = 1.1:0.1:2 ;
AIRtime = []; 
EPIRtime = [] ;

for stm =1:length(mu)
  optionsP.mu = mu(stm) ;
  upalpha = sqrt(mu(stm)/(mu(stm)+2)); 
  adalpha = 5e-2:0.1:upalpha ;
  AIR = MC_AIRNN(X0,Y,sp, lambda, mask, tol, optionsP);
  AIRtime(stm) = AIR.time; 
  options.mu = mu(stm) ;
  for i = 1:length(adalpha)
    options.alpha = adalpha(i);  
    sol = MC_EPIRNN(X0,Y,sp, lambda, mask, tol, options); 
    EPIRtime(stm,i) = sol.time;
  end
end
%%
EPIRtime - AIR.time
%% plot the error and Objf
PIRx = (1:1:PIR.iterTol)/100; AIRx = (1:1:AIR.iterTol)/100; EPIRx = (1:1:EPIR.iterTol)/100; 
% ------------ RelErr plot 
figure(1)
plot(PIRx,log10(PIR.RelErr),'-r','linewidth',1);hold on
plot(AIRx,log10(AIR.RelErr),'--b','linewidth',1);
plot(EPIRx,log10(EPIR.RelErr),'-.g','linewidth',1);hold off
title("RelErr "); 
xlabel("iteration"); ylabel("RelErr")
legend("PIRNN","AIRNN","EPIRNN")

% ------------ RelDist plot 
figure(2)
plot(PIRx,log10(PIR.RelDist),'-r','linewidth',1);hold on
plot(AIRx,log10(AIR.RelDist),'--b','linewidth',1);
plot(EPIRx,log10(EPIR.RelDist),'-.g','linewidth',1);hold off
title("RelDist "); 
xlabel("iteration"); ylabel("RelDist")
legend("PIRNN","AIRNN","EPIRNN")
% % ------------ obj plot 
figure(3)
plot(PIRx,PIR.f,'-r');hold on; plot(AIRx,AIR.f,':k');
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
plot(PIRx,PIR.rank,'-r');hold on; plot(AIRx,AIR.rank,'--b');
plot(EPIRx,EPIR.rank,'-.g');hold off 
title("rank of iterations")
xlabel("iteration"); ylabel("rank")
legend("PIRNN","AIRNN","EPIRNN")
% ------------  time plot
figure(5)

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



