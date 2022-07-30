%% the different method for solving the matric recovery
% step-1 search best lambda with same p for different method
% step-2 compare different restore pic with PSNR 

clear;clc; format long; 
rng(22);
% cwd = fileparts(pwd) ;
path_lena = strcat(pwd,'\img_image\lena.png');
path_re1 = strcat(pwd,'\img_image\re1.jpg');

img_ori = double(imread(path_lena))/255 ; 
% img_ori = double(imread(path_lena))/255;
img_size = size(img_ori);

  %% strictly low rank
  rt = ceil(min(size(img_ori(:,:,1)))/5); 
  for i=1:3
    [U,S,V]=svd(img_ori(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end
  %% mask
    %% random mask
    missrate = 0.3; % sampleRate = 1 - missRate
    mask = zeros(img_size(1:2));
    for i=1:img_size(2)  
        idx = 1:1:img_size(1) ;
        randidx = randperm(img_size(1),img_size(1)); % 随机[n] 中的 k 个 index
        mask(randidx(1:ceil(img_size(1)*missrate)),i)=1; 
    end
    mask = ~mask;
    %% block_column mask 
    mask_path = strcat(cwd,'\img_mask\block_column.bmp');
    omega = double(imread(mask_path))/255.0;
    omega = imresize(omega,img_size(1:2));
    mask = omega(:,:,1);
    %% EN mask 
    mask_path = strcat(cwd,'\img_mask\block_square.bmp');
    omega = double(imread(mask_path))/255.0;
    omega = imresize(omega,img_size(1:2));
    mask = omega(:,:,1);
  %% ------------------------ RECOVERY ------------------------
  % 初始矩阵 + 线性参数 + 正则参数
  XM = mask.*Xt;
  X_INIT_0 = zeros(img_size(1:2)) ; 
  X_INIT_RAND1 = (1+randn(img_size(1),rt))*(randn(rt,img_size(2)));
  tol = 1e-5 ;
  options.max_iter = 5e3; 
  options.eps = 1e-10; 
  options.mu = 1.3;
%   options.KLopt = 1e-5;   
    %% search p
    optionsEP = options; 
    optionsEP.eps = 1 ; optionsEP.Scalar = 0.3 ;
    optionsEP.alpha = 0.7;
%     optionsEP.KLopt = 1e-5;  
    lambda = 0.3;
    SOL_PSNR = zeros(34,1) ; 
    img_Rsol = cell(34,1) ; 
    for pidx = 1:34
      op = 0.01 + (pidx-1)*0.03;
      for channel = 1:3        
        Xm = XM(:,:,channel) ; 
        EPIR = MC_EPIRNN(X0,Xm,op, lambda, mask, tol, optionsEP);
        img_Rsol{pidx}(:,:,channel) = EPIR.Xsol;
      end
      SOL_PSNR(pidx) = psnr(img_ori,img_Rsol{pidx});
    end
    % the best performance of p 
    [~,optSpidx] = max(SOL_PSNR) ; 
    sp =0.3 ; 
    %% search lambda
    Lambda = [1e-5:3e-5:1e-4, 1e-4:3e-4:1e-3, ...
      1e-3:3e-3:1e-2, 1e-2:3e-3:1e-1, ...
      1e-1:3e-1:1, 1:3:10];
    SOL_PSNR = zeros(size(Lambda,2),1); 
    img_Rsol = cell(size(Lambda,2),1) ; 
    for idx_lambda = 1:size(Lambda,2)
       for channel = 1:3        
        Xm = XM(:,:,channel) ; 
        EPIR = MC_EPIRNN(X0,Xm,sp, Lambda(idx_lambda), mask, tol, optionsEP);
        img_Rsol{idx_lambda}(:,:,channel) = EPIR.Xsol;
      end
      SOL_PSNR(idx_lambda) = psnr(img_ori,img_Rsol{idx_lambda});
    end
    [~,optLambdaIdx] = max(SOL_PSNR);
    %% 
    sp = 0.3 ; 
    lambda = 0.1 ; 
    %%
  % PIRNN
  optionsP = options; 
  % AIRNN
  optionsA = options; optionsA.Scalar = 0.1; optionsA.eps = 1;
  % EPIRNN
  optionsEP = optionsA; optionsEP.alpha = 0.7;
  Parsol = {}; 
      %% PIRNN 
  for i =1:3
    Xm = XM(:,:,i);
    PIR = MC_PIRNN(X_INIT_RAND1,Xm,sp, lambda, mask, tol, optionsP);
    X_PIR(:,:,i) = PIR.Xsol; Parsol{i,1} = PIR;
  end
%       %% AIRNN
  for i=1:3
    Xm = XM(:,:,i);
    AIR = MC_AIRNN(X_INIT_RAND1,Xm,sp, lambda, mask, tol, optionsA);
    X_AIR(:,:,i) = AIR.Xsol; Parsol{i,2} = AIR;
  end
%       %% EPIRNN
  for i=1:3
    Xm = XM(:,:,i);
    EPIR = MC_EPIRNN(X_INIT_RAND1,Xm,sp, lambda, mask, 1e-5, optionsEP);
    X_EPIR(:,:,i) = EPIR.Xsol; Parsol{i,3} = EPIR;
  end
      %% SCP ADMM 
  for i=1:3
    Xm = XM(:,:,i);
    labmda = 1 ;
    optionsEP.max_iter = 1e2 ;  
%     lambda = norm(Xm,"fro")*1;
    SCP = MC_SCpADMM(Xm,Xm,sp, lambda, mask, tol, optionsEP);     
    X_SCP(:,:,i) = SCP.Xsol; Parsol{i,4} = SCP;
  end
      %% FGSRp 
  optionsFGSR.p = sp ; 
  optionsFGSR.d = ceil(1.5*rt);
  optionsFGSR.regul_B = "L2";
  optionsFGSR.tol = 1e-4;
  optionsFGSR.lambda = 0.09;
  for i = 1:3
    Xm = XM(:,:,i);
    Xr = MC_FGSRp_PALM(Xm,mask,optionsFGSR);
    X_FGSR(:,:,i) = Xr;
  end
  %% imshow show 
imshow(X_EPIR)









































