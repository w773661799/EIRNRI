rng(22);format long
cwd = fileparts(pwd) ;
path_lena = strcat(cwd,'\img_image\lena.png');
path_re1 = strcat(cwd,'\img_image\re1.jpg');
img_ori = double(imread(path_lena))/255 ; 
% img_ori = double(imread(path_lena))/255;
img_size = size(img_ori);


  %% strictly low rank
  rt = ceil(min(size(img_ori(:,:,1)))/5); 
  for i=1:3
    [U,S,V]=svd(img_ori(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end
  
%% ------------------------ RECOVERY ------------------------
% \lambda ~ [7e-3,50];
% 
  % 初始矩阵 + 线性参数 + 正则参数
  XM = mask.*Xt;
  X_INIT_0 = zeros(img_size(1:2)) ; 
  X_INIT_RAND1 = 1+(1+randn(img_size(1),rt))*(randn(rt,img_size(2)));
  tol = 1e-5 ;
  options.max_iter = 1e2; 
  options.mu = 3.3;  
  options.eps = 1;
  options.KLopt = 1e-5;
  sp = 0.25; 
  lambdaRho = 0.65 ;
  lambdaTol = 5e-3;
%   options.KLopt = 1e-5;   

    %% different method with warm start to solve the MC
% iterTOTAL.pir={}; iterTOTAL.air={}; iterTOTAL.epir={};      
% timeTOTAL.pir={}; timeTOTAL.air={}; timeTOTAL.epir={};
% OBJF.pir={}; OBJF.air={}; OBJF.epir={};
iterTOTAL.pir=zeros(1,3); timeTOTAL.pir={}; OBJF.pir={};
iterTOTAL.air=zeros(1,3); timeTOTAL.air={}; OBJF.air={}; 
iterTOTAL.epir=zeros(1,3); timeTOTAL.epir={}; OBJF.epir={};
%%



%% PIR
  optionsP = options;
  optionsP.eps = 1e-10;
  for i=1:3
    iter = 1;
    Xm = XM(:,:,i);
%     lambda = min(100,norm(Xm)*1e-1);
    lambda = min(norm(Xm)*1e-1,50) ;
    lambda_INIT = max(norm(Xm,"fro")*1e-5,lambdaTol);
    optionsP.max_iter = 2e2;
    X_INIT_PIR = X_INIT_RAND1; 
%     while lambda>lambda_INIT
%       PIR = MC_PIRNN(X_INIT_PIR,Xm,sp, lambda, mask, tol, optionsP);
%       X_INIT_PIR = PIR.Xsol; 
%       lambda = lambda*lambdaRho; 
% %         optionsP.eps =   optionsP.eps * 1e-2; 
%       iterTOTAL.pir(i) = iterTOTAL.pir(i) + PIR.iterTol;
%       if iter==1
%         timeTOTAL.pir{i} = PIR.time;
%         OBJF.pir{i} = PIR.f;
%       else
%         timeTOTAL.pir{i} = [timeTOTAL.pir{i},timeTOTAL.pir{i}(end)+PIR.time];
%         OBJF.pir{i} = [OBJF.pir{i},PIR.f];
%       end
%       iter = iter+1;
%     end
    lambda = norm(Xm,"fro")*1e-3;
    optionsP.eps = 1e-16;
    optionsP.max_iter = 2e3; %optionsP.eps = 1e-5; 
    PIR = MC_PIRNN(Xm,Xm,sp, lambda, mask, tol, optionsP);
    X_PIR(:,:,i) = PIR.Xsol;
    
    iterTOTAL.pir(i) = iterTOTAL.pir(i) + PIR.iterTol;
    if iter==1
      timeTOTAL.pir{i} = PIR.time;
      OBJF.pir{i} = PIR.f;
    else
      timeTOTAL.pir{i} = [timeTOTAL.pir{i},timeTOTAL.pir{i}(end)+PIR.time];
      OBJF.pir{i} = [OBJF.pir{i},PIR.f];
    end
    disp(" ----------------------------------------------------- PIR ")
  end

%%
% AIR
  optionsA = options;
  optionsA.eps = 10; 
  optionsA.Scalar = 0.8;
  
  for i=1:3
    iter = 1;
    Xm = XM(:,:,i); 
    lambda = min(norm(Xm),50) ;
    lambda_INIT = max(norm(Xm,"fro")*1e-5,lambdaTol);
    optionsA.max_iter = 1e2;
    X_INIT_AIR = X_INIT_RAND1; 
%     while lambda>lambda_INIT
%       AIR = MC_AIRNN(X_INIT_AIR,Xm,sp, lambda, mask, tol, optionsA);
%       X_INIT_AIR = AIR.Xsol;
%       lambda = lambda*lambdaRho;
% %       optionsA.eps = optionsA.eps*optionsA.Scalar; 
%       iterTOTAL.air(i) = iterTOTAL.air(i) + AIR.iterTol;
%       if iter==1
%         timeTOTAL.air{i} = AIR.time;
%         OBJF.air{i} = AIR.f;
%       else
%         timeTOTAL.air{i} = [timeTOTAL.air{i},timeTOTAL.air{i}(end)+AIR.time];
%         OBJF.air{i} = [OBJF.air{i},AIR.f];
%       end
%       iter = iter+1;
%     end
    lambda = norm(Xm,"fro")*1e-3;
% lambda = 1;
    optionsA.max_iter = 2e3;
    AIR = MC_AIRNN(Xm, Xm, sp, lambda, mask, tol, optionsA);
    X_AIR(:,:,i) = AIR.Xsol; 
    
    iterTOTAL.air(i) = iterTOTAL.air(i) + AIR.iterTol;
    if iter==1
      timeTOTAL.air{i} = AIR.time;
      OBJF.air{i} = AIR.f;
    else
      timeTOTAL.air{i} = [timeTOTAL.air{i},timeTOTAL.air{i}(end)+AIR.time];
      OBJF.air{i} = [OBJF.air{i},AIR.f];
    end
    iter = iter+1;
    disp(" ----------------------------------------------------- AIR ")
  end

%%
% EPIRNN
  
  optionsEP = options;
  optionsEP.eps = 1e1;
  optionsEP.Scalar = 0.3; % optionsEP = optionsA;
  optionsEP.alpha = 0.7; 
  
  for i=1:3
    iter = 1; 
    Xm = XM(:,:,i); 
    lambda = min(norm(Xm),50) ;
    lambda_INIT = max(norm(Xm,"fro")*1e-3,lambdaTol);
    optionsEP.max_iter = 2e2;
%     X_INIT_EPIR = X_INIT_RAND1; 
%     while lambda>lambda_INIT
%       EPIR = MC_EPIRNN(X_INIT_EPIR,Xm,sp, lambda, mask, tol, optionsEP);
%       X_INIT_EPIR = EPIR.Xsol;
%       lambda = lambda*lambdaRho
%       
%       iterTOTAL.epir(i) = iterTOTAL.epir(i) + EPIR.iterTol;
%       if iter==1
%         timeTOTAL.epir{i} = EPIR.time;
%         OBJF.epir{i} = EPIR.f;
%       else
%         timeTOTAL.epir{i} = [timeTOTAL.epir{i},timeTOTAL.epir{i}(end)+EPIR.time];
%         OBJF.epir{i} = [OBJF.epir{i},EPIR.f];
%       end
%       iter = iter+1;
%     end
    
    lambda = norm(Xm,"fro")*1e-3;optionsEP.eps = 1e-3;
    optionsEP.max_iter = 2e3;
    EPIR = MC_EPIRNN(Xm,Xm,sp, lambda, mask, tol, optionsEP);
    X_EPIR(:,:,i) = EPIR.Xsol; 
      iterTOTAL.epir(i) = iterTOTAL.epir(i) + EPIR.iterTol;
      if iter==1
        timeTOTAL.epir{i} = EPIR.time;
        OBJF.epir{i} = EPIR.f;
      else
        timeTOTAL.epir{i} = [timeTOTAL.epir{i},timeTOTAL.epir{i}(end)+EPIR.time];
        OBJF.epir{i} = [OBJF.epir{i},EPIR.f];
      end
      iter = iter+1;
    disp(" ----------------------------------------------------- EPIR ")
  end

%%
% IRNN 2014 / 2016 Canyi Lu
% parameter should be to the same

%% 
% ADMM SCP

max_iter = 1e2;
opt.omega = mask;
opt.tau = 30;

rou = 1.5;

scpStep = 0.03;
scpMaxLen = floor(1/scpStep); 
for idx=1:scpMaxLen
  opt.p = scpStep*idx;
  for channel = 1:3
    opt.lambda = norm(XM(:,:,channel),"fro")*1e-2;
    opt.D_omega = XM(:,:,channel);
    Y_omega = opt.D_omega;
    E_omega = opt.D_omega;
    W = opt.D_omega;
    Z = Y_omega;
    mv = 1.5;
    for iter = 1:max_iter
        X1 = opt_X(E_omega,Y_omega,W,Z,mv,opt);
        E_omega = opt_E(X1,Y_omega,mv,opt);
        W = opt_W(X1,Z,mv,opt);
        Y_omega = Y_omega + mv*(E_omega - X1 .* opt.omega + opt.D_omega); 
        Z = Z + mv*(X1 - W);
        mv = rou*mv;
    end
    X_SCP(:,:,channel) = X1;
  end
  peak_snr(idx) = psnr(img_ori,X_SCP);
end
%%
optionsSCP.max_iter = 2e3;
optionsSCP.tau = 30;
% [~,scpBestIdx] = max(peak_snr); opt.p = scpStep*scpBestIdx;
for channel = 1:3
  Xm = XM(:,:,channel);
  lambda = norm(Xm,"fro")*1e-3 ; 
  SCP = MC_SCpADMM(Xm,sp,lambda, mask, tol, optionsSCP); 
  X_SCP(:,:,channel) = SCP.Xsol;
end
%% 
subplot(1,2,1);imshow(XM);subplot(1,2,2);imshow(X_SCP)














