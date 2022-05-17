function Par = MC_PIRNN(X0,M,sp, lambda, mask, tol, options)
  % - M is the observation matrix
  % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
  % - mu - the augmented lagrangian parameter, default = 10*lambda
  % - tol - reconstruction error tolerance, default = 1e-6
  % - max_iter - maximum number of iterations, default = 1000
  
  if isfield(options,'max_iter')==0,max_iter = 5e3;
  else,max_iter = options.max_iter ;
  end
  
  if isfield(options,'eps')==0,epsre = 1e-3;
  else,epsre = options.eps ;
  end
  
  if isfield(options,'mu')==0,mu = 1.1;
  else,mu = options.mu ;
  end
  
  if isfield(options,'KLopt')==0,KLopt = 1e-5*min(size(M));
  else,KLopt = options.KLopt ;
  end
  
  if isfield(options,'Rel')==0
    disp("Calculation of correlation distance...");
  else
    ReX = options.Rel; 
    spRelErr = -ones(max_iter,1);
  end
  
  spRelDist = -ones(max_iter,1); spf = -ones(max_iter,1);
  sprank = -ones(max_iter,1);
  Ssim = zeros(max_iter,1); Rsim = zeros(max_iter,1);
  [nr,nc] = size(M) ; rc = min(nr,nc) ;
  weps = ones(rc,1)*epsre;
  Gradf = @(X)(mask.*(X-M)) ; 
  Objf = @(x)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svds(x,rank(x)),sp)^(sp)); 
  ALF = @(x,y)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svd(x)+y,sp)^(sp));
  iter = 0 ;
  sigma = svd(X0); % ch1

  while iter <= max_iter  
    iter = iter + 1; 
%     [ud,sigma,vd] = svd(X0);
%     Rk = rank(X0) ; 
%     RelErr = norm(ud(1:Rk,:)*Gradf(X0)*vd(1:Rk,:)'+...
%       lambda*sp*spdiags(diag(sigma).^(sp-1),0,Rk,Rk),'fro')/norm(M,'fro'); 
  
    [U,S,V] = svd(X0 - Gradf(X0)/mu,'econ') ;
    NewS = diag(S) - 2*lambda*sp*(sigma+weps).^(sp-1)/mu ;
    idx = NewS>eps(1); Rk = sum(idx);  
    X1 = U*spdiags(NewS.*idx,0,rc,rc)*V';
    
    sigma = sort(NewS.*idx,'descend'); % update the sigma 
%     sigma = NewS.*idx ;
    RelDist = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(M,'fro'); 
%     RelErr = norm(X1-M,'fro')/norm(M,'fro');
    KLdist = norm(X1-X0,"fro");
% The Initialization Information
    if iter==1
      fprintf(1, 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
              iter, RelDist, rank(X1),Objf(X1) );
    end
% save for plot 
    spRelDist(iter) = RelDist; spf(iter) = ALF(X0,weps);
    sprank(iter) = rank(X1);
    Rsim(iter) = (Objf(X1)-Objf(X0))/(norm(X1-X0,'fro')^2); 
    Ssim(iter) = norm(U(:,idx)'*Gradf(X1)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(X1-X0,'fro'); 
    GMinf(iter) = norm(Gradf(X1),inf);

% Optimal Condition    
    if exist('ReX','var')
      Rtol = norm(X1-ReX,'fro')/norm(ReX,'fro');
      Rate(iter) = norm(mask.*(X1-ReX),'fro')/norm(mask.*(X0-ReX),'fro');
      spRelErr(iter) = Rtol; 
      if Rtol<=tol
        disp('Satisfying the optimality condition:Relative error'); 
        fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelErr, rank(Xc),Objf(Xc));
        break;  
      end
    end
    
    if RelDist<=tol
      disp('Satisfying the optimality condition:Relative Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1));
      break
    end
    
    if KLdist<=KLopt
      disp("Satisfying  the KL optimality condition"); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X1),Objf(X1))
      break
    end 
%     if iter==max_iter
%       disp("Reach the MAX_ITERATION");
%       fprintf( 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
%         iter, RelDist, rank(X1),Objf(X1) );
%       break
%     end

% update the iteration     
    X0 = X1;  
  end % end while   

  if exist('ReX','var')
    Par.RelErr = spRelErr(1:iter);
    Par.Rate = Rate;
  end
  Par.RelDist = spRelDist(1:iter);
  Par.Obj = Objf(X1); 
  Par.f = spf(1:iter) ;
  Par.rank = sprank(1:iter); Par.iterTol = iter ;
  Par.S = Ssim; Par.R = Rsim;
  Par.Xsol = X1; Par.GMinf = GMinf;
  Par.KLdist = KLdist;
end