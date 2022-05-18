function Par = MC_EPIRNN(X0,M,sp, lambda, mask, tol, options)
  % - M is the observation matrix
  % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
  % - mu - the augmented lagrangian parameter, default = 10*lambda
  % - tol - reconstruction error tolerance, default = 1e-6
  % - max_iter - maximum number of iterations, default = 5e3
  
  if isfield(options,'max_iter')==0,max_iter = 5e3;
  else,max_iter = options.max_iter ;
  end
  
  if isfield(options,'eps')==0,epsre = 1;
  else,epsre = options.eps ;
  end
  
  if isfield(options,'mu')==0,mu = 1.1; % proximal parameter
  else,mu = options.mu ;
  end
  
  if isfield(options,'KLopt')==0,KLopt = 1e-5*min(size(M));
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
  
  spRelDist = []; spf = [];
	sprank = [];
  Ssim = []; Rsim = [];   
  [nr,nc] = size(M); rc = min(nr,nc); 
  weps = ones(rc,1)*epsre; 
  
  Gradf = @(X)(mask.*(X-M)) ; 
  Objf = @(x)(norm(mask.*(x-M),'fro')^2/2 + lambda*norm(svds(x,rank(x)),sp)^(sp));
  ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));
  
  iter = 0; %Par.f = Objf(X0);  
  X1 = X0; 
  sigma = svd(X1); 
  
  tic;
  while iter <= max_iter 
    iter = iter + 1 ; 
    Xc = X1 + alpha*(X1-X0);  
    [U,S,V] = svd(Xc - Gradf(Xc)/mu,'econ') ;
% restart the weps
%     if ~isempty(find(and(diag(S)>zero,(sigma+weps)<zero),1)) && (iter<=1e2)
%       weps(and(diag(S)>zero,(sigma+weps)<zero)) = epsre;
%     end 
    
    NewS = diag(S) - 2*lambda*sp*(sigma+weps).^(sp-1)/mu; 
%     NewS = diag(S) - 2*lambda*sp*(diag(S)+weps).^(sp-1)/mu; 
    NewS(isinf(NewS))=0;
%     NewS(isnan(NewS))=0;
    idx = NewS>zero; Rk = sum(idx);
    Xc = U*spdiags(NewS.*idx,0,rc,rc)*V'; 
%     weps = weps.*(NewS>zero)*Scalar + weps.*(NewS<=zero) ;
    weps(weps(1:Rk)>zero) = weps(weps(1:Rk)>zero)*Scalar ; 
%     weps = [weps(1:Rk)*Scalar; weps(Rk+1:end)]; 
    sigma = sort(NewS.*idx,'descend') ;% update the sigma 
%     sigma = NewS.*idx ;
    RelDist = norm(U(:,idx)'*Gradf(Xc)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(M,'fro');
    KLdist = norm(Xc-X1,"fro")+(1-Scalar)*norm(weps(1:Rk),1)/Scalar;

% save for plot 
    spRelDist(iter) = RelDist; 
    spf(iter) = Objf(Xc);
    Stime(iter) = toc; % recored the computing time 
    sprank(iter) = rank(Xc);
    Rsim(iter) = (Objf(Xc)-Objf(X1))/(norm(Xc-X1,'fro')^2); 
    Ssim(iter) = norm(U(:,idx)'*Gradf(Xc)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(Xc-X1,'fro');    
    GMinf(iter) = norm(Gradf(Xc),inf);
%     sprKLdist(iter) = KLdist;

% The Initialization Information
% disp(RelDist)
%     if iter==1
      %       fprintf(1, 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
%               iter, RelDist, rank(X1),Objf(X1) );
%     end
    
% Optimal Condition 

    if exist('ReX','var')
      Rtol = norm(Xc-ReX,'fro')/norm(ReX,'fro');
      Rate(iter) = norm(mask.*(Xc-ReX),'fro')/norm(mask.*(X1-ReX),'fro');
      spRelErr(iter) = Rtol; 
      if Rtol<tol
        disp('Satisfying the optimality condition:Relative error'); 
        fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelDist, rank(Xc),Objf(Xc))
        break;
      end
      
    end
    
    if norm(mask.*(Xc-M),inf)<tol
      disp("Iteration terminates");
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelDist, rank(Xc),Objf(Xc));
      break;
    end
    
    if RelDist<tol
      disp('Satisfying the optimality condition:Relative Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(Xc),Objf(Xc))
      break
    end

    
    if KLdist<KLopt
      disp("Satisfying  the KL optimality condition"); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(Xc),Objf(Xc))
      break
    end
    
    if iter>max_iter
      disp("Reach the MAX_ITERATION");
      fprintf( 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(Xc),Objf(Xc) );
      break
    end

% update the iteration
    X0 = X1; X1 = Xc; 
  end  % end while 
  estime = toc; 

  if exist('ReX','var')
    Par.RelErr = spRelErr(1:iter); 
    Par.Rate = Rate;
  end

  Par.weps = weps(Rk);
  Par.time = Stime; 
  Par.RelDist = spRelDist(1:iter); 
  Par.Obj = Objf(Xc); 
  Par.f = spf(1:iter) ;
  Par.rank = sprank(1:iter); 
  Par.iterTol = iter ;
  Par.S = Ssim; Par.R = Rsim;
  Par.Xsol = Xc; 
  Par.GMinf = GMinf;
  Par.KLdist = KLdist;
end