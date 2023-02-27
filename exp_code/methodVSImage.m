%% the different method for solving the matric recovery
% step-1 search best lambda with same p for different method
% step-2 compare different restore pic with PSNR 

clear;clc; format long; 
rng(22);
cwd = fileparts(pwd);
path_lena = strcat(cwd,'\img_image\lena.png');
path_re1 = strcat(cwd,'..\img_image\re1.jpg');

img_ori = double(imread(path_lena))/255 ; 
% img_ori = double(imread(path_lena));
img_size = size(img_ori);
% parpol = parpool(16);
  %% strictly low rank
%   rt = ceil(min(size(img_ori(:,:,1)))/5);
RT = [20,25,30,35,40,45];
for iter_rt = 1:1
  rt = RT(iter_rt);
  for i=1:3
    [U,S,V]=svd(img_ori(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%% mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% random mask
  if exist('it_mask','var')==0 || it_mask == 1
    missrate = 0.3; % sampleRate = 1 - missRate
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
  options.eps = 1e-2; 
  options.mu = 1.1;
  
  optionsEP = options; 
  optionsEP.eps = 1;

  optionsScp.max_iter = 5e2;    
  optionsFGSR;
  %%
  scan.p = 1; scan.lambda = 1;
%   options.KLopt = 1e-5;   
  %% ------------------% search lambda -----------------------------
  if exist('scan','var') && isfield(scan,'lambda') && scan.lambda == 1 
    sp = 0.5;
    Lambda = 2.^-(-3:1:11);
    SOL_PSNR = zeros(size(Lambda,2),3); 

    parfor idx_lambda = 1: size(Lambda,2)
      for channel = 1:1
        Xm = XM(:,:,channel) ; 
        EPIR = ds_EPIRNN(Xm,Xm,sp, Lambda(idx_lambda), mask, tol, optionsEP);
      
        SCP = MC_SCpADMM(Xm,Xm,sp, Lambda(idx_lambda), mask, tol, optionsScp);

        Xr = MC_FGSRp_PALM(Xm,mask,sp,Lambda(idx_lambda),optionsFGSR);
        img_Rsol = {EPIR.Xsol, SCP.Xsol, Xr.Xsol};
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
    lambda_fgsr = 1e-2; 
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

        Xr = MC_FGSRp_PALM(Xm,mask,op,lambda_fgsr,optionsFGSR);

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
    sp_ir = 0.3;
    sp_scp = sp_ir;    
    sp_fgsr = sp_ir;
  end

    %% Recovery 
  % PIRNN
  optionsP = options; 

  % AIRNN
  optionsA = optionsP; 
  optionsA.mu = 0.7; 
  optionsA.eps = 1e-3;
  
  % EPIRNN
  optionsEP = optionsA; 
  optionsEP.alpha = 0.7;

  img_show.ori_img = img_ori;
  img_show.low_img = Xt;
  img_show.mask_img = XM;
  Parsol = {}; 

      %% PIRNN
  for i =1:3
    Xm = XM(:,:,i);
    PIR = ds_ProxIRNN(Xm,Xm,sp_ir, lambda_ir, mask, tol, optionsP);
    X_PIR(:,:,i) = PIR.Xsol; 
  end
  Parsol{1} = X_PIR;
%       %% AIRNN
  for i=1:3
    Xm = XM(:,:,i);
    AIR = ds_AdaIRNN(Xm,Xm,sp_ir, lambda_ir, mask, tol, optionsA);
    X_AIR(:,:,i) = AIR.Xsol; 
  end
  Parsol{2} = X_AIR;
%       %% EPIRNN
  for i=1:3
    Xm = XM(:,:,i);
    EPIR = ds_EPIRNN(Xm,Xm,sp_ir, lambda_ir, mask, tol, optionsEP);
    X_EPIR(:,:,i) = EPIR.Xsol;
  end
  Parsol{3} = X_EPIR;
      %% SCP ADMM 
  for i=1:3
    Xm = XM(:,:,i);
    Scp_tau = 10;
    optionsScp.max_iter = 1e3 ;  
%     lambda = norm(Xm,"fro")*1;
    SCP = MC_SCpADMM(Xm,Xm,sp_scp, lambda_scp, mask, tol, optionsScp);
    X_SCP(:,:,i) = SCP.Xsol;
  end
  Parsol{4} = X_SCP;
      %% FGSRp 0.5 
  optionsFGSR.d = ceil(1.5*rt);
  optionsFGSR.regul_B = "L2";
  optionsFGSR.tol = tol;
  lambda_fgsr = 1;
  sp_fgsr = 0.5; 
  for i = 1:3
    Xm = XM(:,:,i);
    Xr = MC_FGSR_PALM(Xm,mask,optionsFGSR); % for p=0.5 only
%     Xr = MC_FGSR_PALM(Xm,mask,sp_fgsr,lambda_fgsr,optionsFGSR);
    X_FGSR(:,:,i) = Xr.Xsol; 
  end
  Parsol{5} = X_FGSR;
  img_show.sol = Parsol;
end
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





































