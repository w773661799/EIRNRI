function Par = ds_EPIRNN(X0,M,sp, lambda, mask, tol, options)
  % - M is the observation matrix
  % - lambda - regularization parameter, default = 1/sqrt(max(N,M))
  % - beta - the augmented lagrangian parameter, default = 10*lambda
  % - tol - reconstruction error tolerance, default = 1e-6
  % - max_iter - maxibetam number of iterations, default = 5e3
%%  
  if isfield(options,'max_iter')==0,max_iter = 2e3;
  else,max_iter = options.max_iter ;
  end
  
  if isfield(options,'eps')==0,epsre = 1;
  else,epsre = options.eps ;
  end
  
  if isfield(options,'beta')==0,beta = 1.1; % proximal parameter
  else,beta = options.beta ;
  end
  
  if isfield(options,'KLopt')==0,KLopt = 1e-5*min(size(M));
  else,KLopt = options.KLopt ;
  end

  if isfield(options,'mu')==0,mu = 0.5; % mu for eps 
  else,mu = options.mu ;
  end  
  
  if isfield(options,'alpha')==0,alpha = 0.7;
  else,alpha = options.alpha ; % 外推因子
  end

  if isfield(options,'Rel')==0
    disp("Calculation of correlation distance...");
  else
    ReX = options.Rel;
    spRelErr = [];
  end
  
  if isfield(options,'zero')==0,zero = 1e-16;
  else,zero = options.zero;   % thresholding
  end

  if isfield(options,'teps')==0,teps = 1e-16;
  else,teps = options.teps;   % restrict the weighted epsilon for AdaIRNN 
  end
%%  
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
  while iter < max_iter
      iter = iter + 1;
      Xc = X1 + alpha*(X1-X0);
      spf(iter) = Objf(X1);
      [U,S,V] = svd(Xc - Gradf(Xc)/beta,'econ');

      NewS = diag(S) - lambda*sp*(sigma+weps).^(sp-1)/beta;
      idx = NewS>zero; Rk = sum(idx);
      Xc = U*spdiags(NewS.*idx,0,rc,rc)*V';

      weps(weps(1:Rk)>zero) = weps(weps(1:Rk)>zero)*mu;
% restrict the eps
      if isfield(options,"teps")
        weps = (weps<teps) .* teps + (weps>=teps) .* weps;
      end

      sigma = sort(NewS.*idx,'descend') ;% update the sigma

% save for plot    
      sprank(iter) = rank(Xc);
      Stime(iter) = toc; % recored the computing time 

% optional information for the iteration 
%     Rsim(iter) = (Objf(Xc)-Objf(X1))/(norm(Xc-X1,'fro')^2); 
%     Ssim(iter) = norm(U(:,idx)'*Gradf(Xc)*V(:,idx)+...
%       lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(Xc-X1,'fro');    
%     GMinf(iter) = norm(Gradf(Xc),inf);

%% ---------------------- Optimal Condition ----------------------
    if exist('ReX','var')
      Rtol = norm(Xc-ReX,'fro')/norm(ReX,'fro');
      Rate(iter) = norm((Xc-ReX),'fro')/norm((ReX),'fro');
      spRelErr(iter) = Rtol; 
      if Rtol < tol
        disp('EPIRNN: Satisfying the optimality condition:Relative error'); 
        fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelErr, rank(Xc),Objf(Xc))
        break
      end
    end

%     subpartial = lambda*sp*[(weps(idx)+NewS(idx)).^(sp-1);zeros(min(nr,nc)-Rk,1)];
%     RelDist = norm(U'*Gradf(Xc)*V + spdiags(subpartial,0,nr,nc), 'fro')/norm(M,'fro'); 
    RelDist = norm(U(:,idx)'*Gradf(Xc)*V(:,idx)+...
      lambda*sp*spdiags(NewS(idx).^(sp-1),0,Rk,Rk),'fro')/norm(X1,'fro'); 
%       lambda*sp*spdiags((weps(idx)+NewS(idx)).^(sp-1),0,Rk,Rk),'fro')/norm(X1,'fro');
    spRelDist(iter) = RelDist; 
    if RelDist<tol
      disp('EPIRNN: Satisfying the optimality condition:Relative Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(Xc),Objf(Xc))
      break
    end

    KLdist = norm(Xc-X1,inf);
%     KLdist = norm(Xc-X1,"fro")+(1-mu)*norm(weps(1:Rk),1)/mu;
    if KLdist<KLopt
      disp("EPIRNN: Satisfying  the KL optimality condition"); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, KLdist, rank(Xc),Objf(Xc))
      break
    end
   
    if iter==max_iter
      disp("EPIRNN: Reach the MAX_ITERATION");
      fprintf( 'iter:%04d\t rank(X):%d\t Obj(F):%d\n', ...
        iter, rank(Xc),Objf(Xc) );
      break
    end
    X0 = X1; X1 = Xc; % update the iteration
  end  % end while 
  estime = toc; 
% KLdist
%% return the best-lambda, time, iterations, rank, objective,and solution
  if exist('ReX','var')
    Par.RelErr = spRelErr(1:iter); 
    Par.Rate = Rate;
  end
  
  Par.time = Stime; 
  Par.f = spf(1:iter) ;
  Par.rank = sprank(1:iter); 
  Par.RelDist = spRelDist(1:iter); 
  
  Par.Obj = Objf(Xc); 
  Par.Xsol = Xc; 
  Par.iterTol = iter ;

  Par.weps = weps(1:end);
%   Par.S = Ssim; Par.R = Rsim;
%   Par.GMinf = GMinf;
  Par.KLdist = KLdist;
end