function Par = MC_AIRNN(X0,M,sp, lambda, mask, tol, options)
  % - M is the observation matrix
  % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
  % - mu - the augmented lagrangian parameter, default = 10*lambda
  % - tol - reconstruction error tolerance, default = 1e-6
  % - max_iter - maximum number of iterations, default = 1000
 
  if isfield(options,'max_iter')==0,max_iter = 2e3;
  else,max_iter = options.max_iter ;
  end
  
  if isfield(options,'eps')==0,epsre = 1;
  else,epsre = options.eps ;
  end
  
  if isfield(options,'mu')==0,mu = 1.1;
  else,mu = options.mu ;
  end
  
  if isfield(options,'KLopt')==0,KLopt = 1e-5*min(size(M));
  else,KLopt = options.KLopt ;
  end
  
  if isfield(options,'Scalar')==0,Scalar = 0.99;
  else,Scalar = options.Scalar; 
  end
  
  if isfield(options,'Rel')==0
    disp("Calculation of correlation distance...");
  else
    ReX = options.Rel; 
    spRelErr = -ones(max_iter,1); 
  end
  
  if isfield(options,'zero')==0,zero = 1e-20;
  else,zero = options.zero;   % thresholding
  end

  spRelDist = []; spf = [];
  sprank = [];
  Ssim = []; Rsim = [] ; 
  [nr,nc] = size(M); rc = min(nr,nc); 
  weps = ones(rc,1)*epsre; 
  
  Gradf = @(X)(mask.*(X-M)); 
  Objf = @(x)(norm(mask.*(x-M),'fro')^2/2 + lambda*norm(svd(x),sp)^(sp)); 
  ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));
  
  iter = 0; %Par.f = Objf(X0);
  sigma = svd(X0); % ch1
  
  tic ;
  while iter <= max_iter
    iter = iter + 1; 
    [U,S,V] = svd(X0 - Gradf(X0)/mu,'econ') ;
% restart the eps
%     if ~isempty(find(and(diag(S)>zero,(sigma+weps)<zero),1)) && (iter<=1e2)
%       weps(and(diag(S)>zero,(sigma+weps)<zero)) = epsre;
%     end 

    NewS = diag(S) - 2*lambda*sp*(sigma+weps).^(sp-1)/mu;
%     NewS = diag(S) - 2*lambda*sp*(diag(S)+weps).^(sp-1)/mu;
    NewS(isinf(NewS))=0; 
    idx = NewS>zero; Rk = sum(idx); 
    % NewS = NewS.*idx;  
    X1 = U*spdiags(NewS.*idx,0,rc,rc)*V'; 
%     eps = eps.*(NewS>zero)*Scalar + eps.*(NewS<=zero); 
%     weps = [weps(1:Rk)*Scalar;weps(Rk+1:end)]; 
    weps(weps(1:Rk)>zero) = weps(weps(1:Rk)>zero)*Scalar ;
    sigma = sort(NewS.*idx,'descend'); % update the sigma 
%     sigma = NewS.*idx; 
    RelDist = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(M,'fro'); 
    KLdist = norm(X1-X0,"fro")+(1-Scalar)*norm(weps(1:Rk),1)/Scalar;
%     RelErr = norm(X1-M,'fro')/norm(M,'fro'); 
% save for plot 
    spRelDist(iter) = RelDist; 
    spf(iter) = Objf(X1);
    Stime(iter) = toc; % recored the computing time 
    sprank(iter) = rank(X1);
    Rsim(iter) = (Objf(X1)-Objf(X0))/(norm(X1-X0,'fro')^2); 
    Ssim(iter) = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(X1-X0,'fro'); 
    GMinf(iter) = norm(Gradf(X1),inf);

% The Initialization Information
%     if iter==1
%       fprintf(1, 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
%               iter, RelDist, rank(X1),Objf(X1) );
%     end

% Optimal Condition 

    if exist('ReX','var')
      Rtol = norm(X1-ReX,'fro')/norm(ReX,'fro');
      Rate(iter) = norm(mask.*(X1-ReX),'fro')/norm(mask.*(X0-ReX),'fro');
      spRelErr(iter) = Rtol;
      if Rtol<=tol
        disp('Satisfying the optimality condition:Relative error'); 
        fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelDist, rank(X1),Objf(X1));
        break;  
      end
    end
    
    if norm(mask.*(X1-M),inf)<tol
      disp("Iteration terminates");
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelDist, rank(X1),Objf(X1));
    end
    
    if RelDist<tol
      disp('Satisfying the optimality condition:Relative Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1));
      break
    end

    if KLdist<KLopt
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
  if mod(iter,1000)==0
     fprintf( 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1) );
  end
% update the iteration 
    X0 = X1 ;   
  end % end while 
  estime = toc;

  if exist('ReX','var')
    Par.RelErr = spRelErr(1:iter); 
    Par.Rate = Rate;
  end

  Par.weps = weps(1:Rk);
  Par.time = Stime;
  Par.RelDist = spRelDist(1:iter); 
  Par.Obj = Objf(X1); 
  Par.f = spf(1:iter) ;
  Par.rank = sprank(1:iter); 
  Par.iterTol = iter ;
  Par.S = Ssim; Par.R = Rsim;
  Par.Xsol = X1; 
  Par.GMinf = GMinf;
  Par.KLdist = KLdist;
end