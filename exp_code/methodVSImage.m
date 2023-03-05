%% the different method for solving the matric recovery
% step-1 search best lambda with same p for different method
% step-2 compare different restore pic with PSNR 

clear;clc;
format long; 
rng(22);
parpol = parpool(16);

cwd = fileparts(pwd);

path_lena = strcat(cwd,'\img_image\lena.png');
img_ori = double(imread(path_lena))/255 ; 

% path_re1 = strcat(cwd,'\img_image\re1.jpg');
% img_ori = double(imread(path_re1))/255 ; 

% path_sist = strcat(cwd,'\img_image\sistlibrary.jpeg');
% img = double(imread(path_sist))/255 ; 
% for i = 1 :3
% img_ori(:,:,i) = img(:,:,i)';
% end

it_mask = 1;
% 

% img_ori = double(imread(path_lena));
img_size = size(img_ori);
noise = 1e0*randn(img_size(1:2));

%   %%  % % % strictly low rank
%   rt = ceil(min(size(img_ori(:,:,1)))/5);
RT = [15,20,25,30,35,40];
for iter_rt = 1:length(RT)
  rt = RT(iter_rt);
  for i=1:3
    [U,S,V]=svd(img_ori(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%% mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% random mask
  if exist('it_mask','var')==0 || it_mask == 1
    missrate = 0.2; % sampleRate = 1 - missRate
    mask = zeros(img_size(1:2));
    for i=1:img_size(2)  
        idx = 1:1:img_size(1) ;
        randidx = randperm(img_size(1),img_size(1)); % 随机[n] 中的 k 个 index
        mask(randidx(1:ceil(img_size(1)*missrate)),i)=1; 
    end
    mask = ~mask;
  elseif it_mask == 2
    %% block_column mask 
    mask_path = strcat(cwd,'\img_mask\block_column.bmp');
    omega = double(imread(mask_path))/255.0;
    omega = imresize(omega,img_size(1:2));
    mask = omega(:,:,1);
  elseif it_mask == 3
    %% EN mask 
    mask_path = strcat(cwd,'\img_mask\block_square.bmp');
    omega = double(imread(mask_path))/255.0;
    omega = imresize(omega,img_size(1:2));
    mask = omega(:,:,1);
  end
  %% ------------------------ RECOVERY ------------------------
  % 初始矩阵 + 线性参数 + 正则参数
  XM = mask.*Xt;
  X_INIT_0 = zeros(img_size(1:2)) ; 
  X_INIT_RAND1 = (1+randn(img_size(1),rt))*(randn(rt,img_size(2)));
  tol = 1e-5 ;
  max_iter = 1e3;
  Klopt = 1e-7;
  options.max_iter = max_iter; 
%   options.eps = 5e-1; 
  options.mu = 1.1;
  
  optionsEP = options; 
  optionsEP.eps = 1;

  optionsScp.max_iter = 5e2;    
  optionsFGSR.regul_B = "L2";
  %%
%   scan.p = 1; 
  scan.lambda = 1;
%   options.KLopt = 1e-5;   
  %% ------------------% search lambda -----------------------------
  if exist('scan','var') && isfield(scan,'lambda') && scan.lambda == 1 
    sp = 0.5;
%     Lambda = 2.^-(-3:1:11);
    Lambda = [1:0.5:8,2.^(-1:-1:-16)]; 
    SOL_PSNR = zeros(size(Lambda,2),3); 

    parfor idx_lambda = 1: size(Lambda,2)
      for channel = 1:1
        Xm = XM(:,:,channel) ; 
        EPIR = ds_EPIRNN(Xm,Xm,sp, Lambda(idx_lambda), mask, tol, optionsEP);
      
        SCP = MC_SCpADMM(Xm,Xm,sp, Lambda(idx_lambda), mask, tol, optionsScp);

%         FGSR = MC_FGSR_PALM(Xm,mask,Lambda(idx_lambda),optionsFGSR); 
% img_Rsol = {EPIR.Xsol, SCP.Xsol, Xr};

%         Xr = MC_FGSR_PALM(Xm,mask,Lambda(idx_lambda),optionsFGSR); % for p=0.5 only
%         [Xr,~,~] = RPCA_FGSR_ADMM(Xm,Lambda(idx_lambda),optionsFGSR);
        FGSR = MC_FGSRp_PALM(Xm,mask,sp,Lambda(idx_lambda),optionsFGSR);
        img_Rsol = {EPIR.Xsol, SCP.Xsol, FGSR.Xsol};

      end
      SOL_PSNR(idx_lambda,:) = [ psnr(img_ori(:,:,1), img_Rsol{1}), ...
      psnr(img_ori(:,:,1),img_Rsol{2}), psnr(img_ori(:,:,1),img_Rsol{3})];
    end
    [~,optLambdaIdx] = max(SOL_PSNR(:,1)); 
    lambda_ir = Lambda(optLambdaIdx); 
    
    [~,optLambdaIdx] = max(SOL_PSNR(:,2)); 
    lambda_scp = Lambda(optLambdaIdx); 

    [~,optLambdaIdx] = max(SOL_PSNR(:,3)); 
    lambda_fgsr = Lambda(optLambdaIdx); 
  else
    lambda_ir = 1;
    lambda_scp = 1;
    lambda_fgsr = 1; 
  end
  %% ------------------% search p----------------------------- 
  if exist('scan','var') && isfield(scan,'p') && scan.p == 1
    p_step = 34;
    optionsEP = options; 
    optionsEP.eps = 1 ; 
    optionsEP.alpha = 0.7;
    SOLOP_PSNR = zeros(34,3) ; 
%     img_Rsol = cell(34,1) ; 
    parfor pidx = 1: 34
      op = 0.01 + (pidx-1)*0.03;
      for channel = 1:1
        Xm = XM(:,:,channel) ; 
        EPIR = ds_EPIRNN(Xm,Xm,op, lambda_ir, mask, tol, optionsEP);

        SCP = MC_SCpADMM(Xm,Xm,op, lambda_scp, mask, tol, optionsScp);

%         [Xr,~,~] = RPCA_FGSR_ADMM(Xm,lambda_fgsr,optionsFGSR);
%         Xr = MC_FGSRp_PALM(Xm,mask,op,lambda_fgsr,optionsFGSR);
        Xr = MC_FGSRp_PALM(Xm,mask,sp,lambda_fgsr,optionsFGSR);

        img_op_Rsol = {EPIR.Xsol, SCP.Xsol, Xr.Xsol};
      end
      SOLOP_PSNR(pidx,:) = [ psnr(img_ori(:,:,1), img_op_Rsol{1}),...
      psnr(img_ori(:,:,1),img_op_Rsol{2}), psnr(img_ori(:,:,1),img_op_Rsol{3})];
    end
    % the best performance of p 
    [~,optSpidx] = max(SOLOP_PSNR(:,1)); 
    sp_ir =  0.01 + (optSpidx-1)*0.03;
    
    [~,optSpidx] = max(SOLOP_PSNR(:,2)); 
    sp_scp =  0.01 + (optSpidx-1)*0.03;

    [~,optSpidx] = max(SOLOP_PSNR(:,3)); 
    sp_fgsr = 0.01 + (optSpidx-1)*0.03;
  else
    sp_ir = 0.5;
    sp_scp = sp_ir;    
    sp_fgsr = sp_ir;
  end

    %% Recovery 
  % PIRNN
  optionsP = options; 

  % AIRNN
  optionsA = optionsP; 
  optionsA.mu = 0.7; 
  optionsA.eps = 5e-1;
  
  % EPIRNN
  optionsEP = optionsA; 
  optionsEP.alpha = 0.7;
    %%

  img_show.ori_img = img_ori;
  img_show.low_img = Xt;
  img_show.mask_img = XM;
  Parsol = {}; 

      % PIRNN
  for i =1:3
    Xm = XM(:,:,i);
    PIR = ds_ProxIRNN(Xm+noise,Xm,sp_ir, lambda_ir, mask, tol, optionsP);
    X_PIR(:,:,i) = PIR.Xsol; 
  end
  Parsol{1} = X_PIR;
%       %% AIRNN
  for i=1:3
    Xm = XM(:,:,i);
    AIR = ds_AdaIRNN(Xm+noise,Xm,sp_ir, lambda_ir, mask, tol, optionsA);
    X_AIR(:,:,i) = AIR.Xsol; 
  end
  Parsol{2} = X_AIR;
%       %% EPIRNN
  for i=1:3
    Xm = XM(:,:,i);
    EPIR = ds_EPIRNN(Xm+noise,Xm,sp_ir, lambda_ir, mask, tol, optionsEP);
    X_EPIR(:,:,i) = EPIR.Xsol;
  end
  Parsol{3} = X_EPIR;
      %% SCP ADMM 
  for i=1:3
    Xm = XM(:,:,i);
    Scp_tau = 10;
    optionsScp.max_iter = 1e3 ;  
%     lambda = norm(Xm,"fro")*1;
    SCP = MC_SCpADMM(Xm+noise,Xm,sp_scp, lambda_scp, mask, 1e-4, optionsScp);
    X_SCP(:,:,i) = SCP.Xsol;
  end
  Parsol{4} = X_SCP;
      %% FGSRp 0.5 
  optionsFGSR.d = ceil(1.5*rt);
  optionsFGSR.alphda = 1e-1;
  optionsFGSR.max_iter = 1e3;
  optionsFGSR.tol = tol;
%   lambda_fgsr = 1;
%   sp_fgsr = 0.5; 
  for i = 1:3
    Xm = XM(:,:,i);
%     Xr = MC_FGSR_PALM(Xm,mask,lambda_fgsr,optionsFGSR); % for p=0.5 only
%     [Xr,~,~] = RPCA_FGSR_ADMM(Xm,5e-2,optionsFGSR); X_FGSR(:,:,i) = Xr;
    FGSR = MC_FGSRp_PALM(Xm,mask,sp_fgsr,lambda_fgsr,optionsFGSR);
%     FGSR = MC_FGSR_PALM(Xm,mask,lambda_fgsr,optionsFGSR); X_FGSR(:,:,i) = FGSR.Xsol;
%     Xr = MC_FGSR_PALM(Xm,mask,sp_fgsr,lambda_fgsr,optionsFGSR);
%     X_FGSR(:,:,i) = Xr.Xsol; 
    X_FGSR(:,:,i) = FGSR.Xsol;
  end
  Parsol{5} = X_FGSR;
  img_show.sol = Parsol;
  Tab_img{iter_rt} = img_show;
%   save('../')
%   save("..\exp_cache\Rank_Robust.mat","Rank_RC",'-mat')
end
%%
%  save("..\exp_cache\ImgRe_RandMask_R30_R1.mat","img_show",'-mat') 
% save R1 to plot 
% save("..\exp_cache\ImgRe_RandMask_Table_BestLambda_0305.mat","Tab_img",'-mat') 
% save for tbale
  %% imshow show 
subplot(1,5,1)
imshow(X_PIR)

subplot(1,5,2)
imshow(X_AIR)

subplot(1,5,3)
imshow(X_EPIR)

subplot(1,5,4)
imshow(X_SCP)

subplot(1,5,5)
imshow(X_FGSR)
%%
for i =1:5
  PSNR_img(i) = psnr(img_ori,Parsol{i})
end
%%
% save("..\exp_cache\ImgRe_best_lambda_LenaR15 .mat","img_show",'-mat') 




































