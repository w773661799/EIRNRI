X0 = X_INIT_0;
iter = 0;
%%
if isfield(options,'max_iter')==0,max_iter = 5e3;
  else,max_iter = options.max_iter ;
  end
  
  if isfield(options,'eps')==0,epsre = 1;
  else,epsre = options.eps ;
  end
  
  if isfield(options,'mu')==0,mu = 1.1; % proximal parameter
  else,mu = options.mu ;
  end
  
  if isfield(options,'KLopt')==0,KLopt = 1e-5*min([size(M),rank(M)]);
  else,KLopt = options.KLopt ;
  end

  if isfield(options,'Scalar')==0,Scalar = 0.3; % Scalar for eps 
  else,Scalar = options.Scalar ;
  end  
  
  if isfield(options,'alpha')==0,alpha = 0.7;
  else,alpha = options.alpha ; % 外推因子
  end

  if isfield(options,'Rel')==0
    disp("Calculation of correlation distance...");
  else
    ReX = options.Rel;
    spRelErr = -ones(max_iter,1);
  end
  
  if isfield(options,'zero')==0,zero = 0;
  else,zero = options.zero;   % thresholding
  end
  
  spRelDist = -ones(max_iter,1); spf = -ones(max_iter,1);
	sprank = -ones(max_iter,1);
  Ssim = []; Rsim = [];   
  [nr,nc] = size(M); rc = min(nr,nc); 
  weps = ones(rc,1)*epsre; 
  
  Gradf = @(X)(mask.*(X-M)) ; 
  Objf = @(x)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svds(x,rank(x)),sp)^(sp));
  ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));
  reWeights = @(x,y)(x+y).^(sp-1);
  iter = 0;  X1 = X0; sigma = svd(X1); Rk = rc;
  
%   Objf(X0)

  tic;
  % 346
  while iter <= max_iter
    iter = iter + 1 ; 
    Xc = X1 + alpha*(X1-X0);  
    [U,S,V] = svd(Xc - Gradf(Xc)/mu,'econ') ;
% restart the weps
%     if ~isempty(find(and(diag(S)>zero,(sigma+weps)<zero),1)) && (iter<=1e2)
%       weps(and(diag(S)>zero,(sigma+weps)<zero)) = epsre;
%     end 
    NewS = diag(S);
    NewS(1:Rk) = NewS(1:Rk) - 2*lambda*sp*reWeights(NewS(1:Rk),weps(Rk))/mu; 
    NewS(Rk+1:end) = 0 ;
%     NewS(isinf(NewS))=0;
%     NewS(isnan(NewS))=0; 

    idx = NewS>eps(1); Rk = sum(idx);
    Xc = U*spdiags(NewS.*idx,0,rc,rc)*V'; 
%     weps = weps.*(NewS>zero)*Scalar + weps.*(NewS<=zero) ;
    weps(weps(1:Rk)>zero) = weps(weps(1:Rk)>zero)*Scalar ; 
%     weps = [weps(1:Rk)*Scalar; weps(Rk+1:end)]; 
    sigma = sort(NewS.*idx,'descend') ;% update the sigma 
%     sigma = NewS.*idx ;
    AbsDist = norm(mask.*(Xc-M),"fro"); 
    RelDist = norm(U(:,idx)'*Gradf(Xc)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(Xc,'fro');
    KLdist = norm(Xc-X1,"fro")+(1-Scalar)*norm(weps(1:Rk),1)/Scalar;

% The Initialization Information
    if iter==1
      fprintf(1, 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
              iter, RelDist, rank(X1),Objf(X1) );
    end
    
% save for plot 
    spRelDist(iter) = RelDist; spf(iter) = ALF(Xc,weps);
    sprank(iter) = rank(Xc);
    Rsim(iter) = (Objf(Xc)-Objf(X1))/(norm(Xc-X1,'fro')^2); 
    Ssim(iter) = norm(U(:,idx)'*Gradf(Xc)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(Xc-X1,'fro');    
    GMinf(iter) = norm(Gradf(Xc),inf);
    sprKLdist(iter) = KLdist;

% Optimal Condition 
    if AbsDist<=tol
      disp('Satisfying the STOP condition: ABS Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelDist, rank(Xc),Objf(Xc))
    end  

%     if exist('ReX','var')
%       Rtol = norm(Xc-ReX,'fro')/norm(ReX,'fro');
%       Rate(iter) = norm(mask.*(Xc-ReX),'fro')/norm(mask.*(X1-ReX),'fro');
%       spRelErr(iter) = Rtol; 
%       if Rtol<=tol
%         disp('Satisfying the optimality condition:Relative error'); 
%         fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
%           iter, RelDist, rank(Xc),Objf(Xc))
%       end
%       break;
%     end

    if RelDist<=tol 
      disp('Satisfying the optimality condition:Relative Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1))
      break
    end

    
    if KLdist<=KLopt
%       || (iter>1 && abs(sprKLdist(iter-1)-KLdist)/KLdist<=KLopt)
      disp("Satisfying  the KL optimality condition"); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1))
      break
    end
    
    if iter==max_iter
      disp("Reach the MAX_ITERATION");
      fprintf( 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1) );
      break
    end

% update the iteration
    X0 = X1; X1 = Xc; 
  end  % end while 
  estime = toc; 
  sprKLdist = sprKLdist(1:iter);
%% 
clear;clc
rng(22);format short 
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
%% ------------------------ RECOVERY ------------------------
  % 初始矩阵 + 线性参数 + 正则参数
  XM = mask.*Xt;
  X_INIT_0 = zeros(img_size(1:2)) ; 
  X_INIT_RAND1 = (1+randn(img_size(1),rt))*(randn(rt,img_size(2)));
  tol = 1e-5 ;
  options.max_iter = 1e2; 
  options.eps = 1e-10; 
  options.mu = 1.3;
%   options.KLopt = 1e-5;   

    %% search p
    sp = 0.3; 
    optionsP = options;
    optionsEP = options; 
    optionsEP.eps = 1; 
    optionsEP.Scalar = 0.3;
    optionsEP.alpha = 0.7;
%     optionsEP.KLopt = 1e-5;  
%       %% EPIRNN

% PIR
  PIR_time = 0;
  for i=1:3
    Xm = XM(:,:,i); 
    lambda = norm(Xm,"fro")*10;
    optionsP.max_iter = 1e2;
    X_INIT_PIR = X_INIT_RAND1; 
    while lambda>=1e-3
      PIR = MC_PIRNN(X_INIT_PIR,Xm,sp, lambda, mask, tol, optionsP);
      X_INIT_PIR = PIR.Xsol; 
      lambda = lambda*0.1; 
      PIR_time = PIR_time + PIR.time; 
    end
    optionsP.max_iter = 1e3;
    PIR = MC_PIRNN(X_INIT_PIR,Xm,sp, lambda, mask, tol, optionsP);
    PIR_time = PIR_time + PIR.time; 
    X_PIR(:,:,i) = PIR.Xsol; Parsol{i,1} = PIR;
  end

%%
% AIR
  AIR_time = 0 ; 
  for i=1:3
    Xm = XM(:,:,i); 
    lambda = norm(Xm,"fro")*10;
    optionsEP.max_iter = 1e2;
    X_INIT_AIR = X_INIT_RAND1; 
    while lambda>=1e-3
      AIR = MC_AIRNN(X_INIT_AIR,Xm,sp, lambda, mask, tol, optionsEP);
      X_INIT_AIR = AIR.Xsol;
      lambda = lambda*0.1;
      AIR_time = AIR_time + AIR.time;
    end
    optionsEP.max_iter = 1e3;
    AIR = MC_AIRNN(X_INIT_AIR, Xm, sp, lambda, mask, tol, optionsEP);
    X_AIR(:,:,i) = AIR.Xsol; 
    AIR_time = AIR_time + AIR.time;
    Parsol{i,2} = PIR;
  end

%%
% EPIRNN
  EPIR_time = 0;
  for i=1:3
    Xm = XM(:,:,i); 
    lambda = norm(Xm,"fro")*10;
    optionsEP.max_iter = 1e2;
    X_INIT_EPIR = X_INIT_RAND1; 
    while lambda>=1e-3
      EPIR = MC_EPIRNN(X_INIT_EPIR,Xm,sp, lambda, mask, tol, optionsEP);
      X_INIT_EPIR = EPIR.Xsol;
      lambda = lambda*0.1;
      EPIR_time = EPIR_time + PIR.time; 
    end
    optionsEP.max_iter = 1e3;
    EPIR = MC_EPIRNN(X_INIT_EPIR,Xm,sp, lambda, mask, 1e-5, optionsEP);
    EPIR_time = EPIR_time + PIR.time; 
    X_EPIR(:,:,i) = EPIR.Xsol; Parsol{i,3} = EPIR;
  end

%% 
% ADMM SCP
  SCP_time = 0;
  for i=1:3
    Xm = XM(:,:,i); 
    lambda = norm(Xm,"fro")*10;
    optionsEP.max_iter = 1e2;
    X_INIT_SCP = X_INIT_RAND1;
    while lambda>=1e-3
      SCP = MC_SCpADMM(X_INIT_SCP,Xm,sp, lambda, mask, tol, optionsEP);
      X_INIT_SCP = SCP.Xsol;
      lambda = lambda*0.1; 
      SCP_time = SCP_time + SCP.time;
    end
    optionsEP.max_iter = 1e3;
    SCP = MC_SCpADMM(X_INIT_SCP,Xm,sp, lambda, mask, tol, optionsEP);
    SCP_time = SCP_time + SCP.time;
    X_SCP(:,:,i) = SCP.Xsol; Parsol{i,4} = SCP;
  end


















