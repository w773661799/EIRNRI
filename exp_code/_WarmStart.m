%% without Warm START
% START with obsevied points
% the parameters as:
% lambda = norm(M,"fro")*1e-3, sp = 0.25
% the other parameters are the default value

%% matrix completion form the random mask
clear;clc;format long 
start = 1;

while start==1
start=0;

rng(22);format long
% cwd = fileparts(pwd) ;
path_lena = strcat(pwd,'\img_image\lena.png');
path_re1 = strcat(pwd,'\img_image\re1.jpg');
img_ori = double(imread(path_lena))/255 ; 
% img_ori = double(imread(path_lena))/255;
img_size = size(img_ori);

%% mask
  %% random mask. mask = 1 
  SR = 0.5; % sampleRate = 1 - missRate
  mask = zeros(img_size(1:2));
  for i=1:img_size(2)
      idx = 1:1:img_size(1) ;
      randidx = randperm(img_size(1),img_size(1)); % 随机[n] 中的 k 个 index
      mask(randidx(1:ceil(img_size(1)*SR)),i)=1; 
  end
  % mask should obtain (1-missrate)*m*n elements from the original image
%% strictly low rank
% save the top 20% of the singular value
  rt = ceil(max(size(img_ori(:,:,1)))/5); 
  for i=1:3
    [U,S,V]=svd(img_ori(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end

%%% ------------------------ RECOVERY ------------------------
%% Comparative with IRNN 
X0 = randn(img_size(1:2));
XM = mask.*Xt;
tol = 1e-5 ;
% scale_lambda = 2e-3;
mu = 0.1; 
proxbeta = 1.3;
maxIter = 2e3;
options.max_iter = maxIter;  
options.beta = proxbeta;
options.KLopt = tol;
sp = 0.5; 

% PIRNN
optionsP = options; optionsP.eps = 1e-5;

% %% optionsEP = options; 
optionsEP.eps=1e-3;
optionsEP.mu = mu; 
optionsEP.alpha = 0.75; 

  %% search best parameters for PIRNN IRNN EPIRNN
  % search lambda 
  Lambda_search = (1.5:0.1:2.1).*10^(-4);
  Xm = XM(:,:,1);
  for lambda_iter = 1:length(Lambda_search)
    lambda = Lambda_search(lambda_iter)*norm(Xm,"fro");
    EPIR_Lambda = MC_EPIRNN(X0+Xm,Xm,sp, lambda, mask, tol, optionsEP);
    lambda_psnr(lambda_iter) = psnr(Xt(:,:,1),EPIR_Lambda.Xsol); 
  end

  [~,lambda_ldx] = max(lambda_psnr);
  scale_lambda = Lambda_search(lambda_ldx);
  %% find the warmstart pointers with best search lambda 
  for channel=1:3
    Xm = XM(:,:,channel);
    EPIR_XWS = MC_EPIRNN(Xm,Xm,sp, norm(Xm,"fro")*scale_lambda*1e1, mask, tol, optionsEP);
    X_WS(:,:,channel) = EPIR_XWS.Xsol; 
  end
    %% PIRNN
    for channel=1:3
      Xm = XM(:,:,channel); 
      lambda = norm(Xm,"fro")*scale_lambda;
      PIR = MC_PIRNN(X_WS(:,:,channel),Xm,sp, lambda, mask, tol, optionsP);
      imgR.pir(:,:,channel) = PIR.Xsol;
      timeTotal.pir{channel} = PIR.time;
      Objective.pir{channel} = PIR.f;
      iterRank.pir{channel} = PIR.rank;
    end
    disp("---------------------------------- PIRNN")
    %% AIRNN
    optionsA = options; 
    optionsA.eps=1e-3;
    optionsA.mu = mu; 
    for channel=1:3
      Xm = XM(:,:,channel); 
      lambda = norm(Xm,"fro")*scale_lambda;
      AIR = MC_AIRNN(X_WS(:,:,channel),Xm,sp, lambda, mask, tol, optionsA);
      imgR.air(:,:,channel) = AIR.Xsol;
      timeTotal.air{channel} = AIR.time;
      Objective.air{channel} = AIR.f;
      iterRank.air{channel} = AIR.rank;
    end
    disp("---------------------------------- AIRNN")
    %% EPIRNN
    for channel=1:3
      Xm = XM(:,:,channel); 
      lambda = norm(Xm,"fro")*scale_lambda;
      EPIR = MC_EPIRNN(X_WS(:,:,channel),Xm,sp, lambda, mask, tol, optionsEP);
      imgR.epir(:,:,channel) = EPIR.Xsol;
      timeTotal.epir{channel} = EPIR.time;
      Objective.epir{channel} = EPIR.f;
      iterRank.epir{channel} = EPIR.rank;
    end
    disp("---------------------------------- EPIRNN")
    clear scale_lambda lambda_psnr
%% Comparative with SCP ADMM 
optionsSCP.max_iter = 200;
optionsSCP.tau = 30;
% optionsSCP.max_iter = maxIter; %200;
  %% search lambda for SCP 
  % ???????????????????????????????????? lambda 没找到?
  Lambda_SCP = (1:1:9).*10.^(-3);
  Xm = XM(:,:,1);
  for lambda_iter = 1:length(Lambda_SCP)
    lambda = norm(Xm,"fro")*Lambda_SCP(lambda_iter);
    SCP = MC_SCpADMM(Xm, sp, lambda, mask, tol, optionsSCP);
    lambda_psnr(lambda_iter) = psnr(Xt(:,:,1),SCP.Xsol); 
  end
  [~,lambda_ldx] = max(lambda_psnr);
  scale_lambda = Lambda_SCP(lambda_ldx);
  %% SCP ADMM
  lambda_scp = 1; 
  for channel=1:3
    Xm = XM(:,:,channel); 
    lambda = norm(Xm,"fro")*scale_lambda;
    SCP = MC_SCpADMM(Xm, sp, lambda_scp, mask, tol, optionsSCP);
    imgR.scp(:,:,channel) = SCP.Xsol;
    timeTotal.scp{channel} = SCP.time;
    Objective.scp{channel} = SCP.f;
    iterRank.scp{channel} = SCP.rank;
  end
  disp("---------------------------------- SCPADMM")
%% IRNN_Lu 2014
% % % % ??? 啥玩意儿啊, 热启动也不行???
% fun_irnn = 'lp'; 
% optionsIRNN.max_iter = 2e3;
% optionsIRNN.gamma = sp;
% optionsIRNN.tol = tol;
% optionsIRNN.mu = proxbeta;
% m = img_size(1); n = img_size(2);
% M_IRNN = opRestriction(m*n,find(mask==1));
% 
% for channel=1:3
%   Xm = XM(:,:,channel); 
% %   xIR = Xm(:);
% xIR = X_WS(:,:,channel);
%   y = M_IRNN(Xm(:),1);
% %   optionsIRNN.lambda_Init = norm(Xm)*1e0;
%   optionsIRNN.lambda_Target = norm(Xm,"fro")*scale_lambda;
%   optionsIRNN.lambda_rho = 0.98; 
%   Sol_IRNN = IRNN_MCLu(xIR(:), fun_irnn, y, M_IRNN, m, n, optionsIRNN);
%   imgR.irnn(:,:,channel) = Sol_IRNN.Xsol;
%   timeTotal.irnn{channel} = Sol_IRNN.time; 
%   Objective.irnn{channel} = Sol_IRNN.f; 
%   iterRank.irnn{channel} = Sol_IRNN.rank; 
% end
% disp("---------------------------------- IRNN_Lu")
% %% 
% Sol_IRNN = IRNN(fun_irnn,y,M,m,n,0.5,optionsIRNN.lambda_Init,0.98,tol);

%% Comparative with FGSR
  %% search lambda for FGSR
  optionsFGSR.tol=1e-5;
  optionsFGSR.p = sp;
  optionsFGSR.maxiter = maxIter*1e1;
  
  clear scale_lambda lambda_psnr 
  Lambda_FGSR = (1:1:9).*10.^(-1);
  Xm = XM(:,:,1); 
  for lambda_iter = 1:length(Lambda_FGSR)
    optionsFGSR.lambda = norm(Xm,"fro")*Lambda_FGSR(lambda_iter);
    Sol_FGSRP = MC_FGSRp_PALM(Xm,mask,optionsFGSR);
    lambda_psnr(lambda_iter) = psnr(Xt(:,:,1),Sol_FGSRP.Xsol);
  end
  [~,lambda_ldx] = max(lambda_psnr);
  scale_lambda = Lambda_FGSR(lambda_ldx);
  %% FGSR
  for channel=1:3
    Xm = XM(:,:,channel); 
    optionsFGSR.lambda = norm(Xm,"fro")*scale_lambda;
    Sol_FGSRP = MC_FGSRp_PALM(Xm,mask,optionsFGSR);
    imgR.fgsrp(:,:,channel) = Sol_FGSRP.Xsol;
    timeTotal.fgsrp{channel} = Sol_FGSRP.time; 
    iterRank.fgsrp{channel} = Sol_FGSRP.rank; 
  end
  disp("---------------------------------- FGSR")
%% Comparative with niAPG
  %% search lambda for niAPG
  
    %% niAPG
    optionsAPG.tol=1e-4;
    optionsAPG.maxIter = maxIter;
    optionsAPG.regType = 2;
    
    clear scale_lambda lambda_psnr 
    Lambda_APG = (1:1:5).*10^(-5);
    Xm = XM(:,:,1); 
    for lambda_iter =1:length(Lambda_APG)
      lambda = norm(Xm,"fro")*Lambda_APG(lambda_iter);
      theta = sqrt(lambda);
      [u,s,v,~] = APGnc(Xm,lambda,theta,optionsAPG);
      lambda_psnr(lambda_iter) = psnr(Xt(:,:,1),u*s*v');
    end
    [~,lambda_ldx] = max(lambda_psnr);
    scale_lambda = Lambda_APG(lambda_ldx);
    %% 
    for channel=1:3
      Xm = XM(:,:,channel); 
      lambda = norm(Xm,"fro")*scale_lambda;
      theta = sp;
      
      [u_ext,s_ext,v_ext,sol_ext]=APGncext(Xm,lambda,theta,optionsAPG);
      imgR.apgext(:,:,channel) = u_ext*s_ext*v_ext';
      timeTotal.apgext{channel} = sol_ext.Time; 
      iterRank.apgext{channel} = sol_ext.Rank; 
      
      [u,s,v,sol] = APGnc(Xm,lambda,theta,optionsAPG);
      imgR.apg(:,:,channel) = u*s*v';
      timeTotal.apg{channel} = sol.Time; 
      iterRank.apg{channel} = sol.Rank; 
    end
    disp("---------------------------------- APG")

end
%% 
sp = 0.5;

%% plot

% figure("units","normalized","position",[0, 0, 0.4, 0.33])
%   imshow(img_ori,"border","tight","initialmagnification","fit"); 
% figure("units","normalized","position",[0, 0, 0.4, 0.33])  
%   imshow(Xt,"border","tight","initialmagnification","fit");
% figure("units","normalized","position",[0, 0, 0.4, 0.33])  
%   imshow(imgR.pir,"border","tight","initialmagnification","fit");
% figure("units","normalized","position",[0, 0, 0.4, 0.33])  
%   imshow(imgR.air,"border","tight","initialmagnification","fit");
% figure("units","normalized","position",[0, 0, 0.4, 0.33])  
%   imshow(imgR.pir,"border","tight","initialmagnification","fit"); 
figure("units","normalized","position",[0.1, 0.1, 0.8, 0.2])  
  subplot('Position',[0 0.2 0.1 0.8]); imshow(img_ori); 
  hold on; xlabel("(a)")
  subplot('Position',[0.15 0.2  0.1 0.8]); imshow(Xt)
  xlabel("(b)")
  subplot('Position',[0.3 0.2 0.1 0.8]); imshow(imgR.pir)
  xlabel("(c)")
  subplot('Position',[0.45 0.2 0.1 0.8]); imshow(imgR.air)
  xlabel("(d)")
  subplot('Position',[0.6 0.2 0.1 0.8]); imshow(imgR.epir);
  xlabel("(e)");  hold off

%%
timeTotal
