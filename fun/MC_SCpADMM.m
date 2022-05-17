function Par = MC_SCpADMM(X0,M,sp, lambda, mask, tol, options)

  if isfield(options,'max_iter')==0,max_iter = 100;
  else,max_iter = options.max_iter ;
  end

  opt.lambda = lambda; % regularization parameter
  opt.p = sp; % Scp norm
  opt.tau = 30; % Scp norm thresholder 
    % tua=0 means the Schatten-p norm  
  opt.omega = mask; % mask set
  opt.D_omega = M; % observation set
  
  Objf = @(x)(norm(mask.*(x-M),'fro')/2 + lambda*norm(svds(x,rank(x)),sp))^(sp);
  Gradf = @(X)(mask.*(X-M)) ; 
    
  Y_omega = opt.D_omega;
  E_omega = X0-opt.D_omega; % E = X-D 
  W = opt.D_omega;
  Z = Y_omega;
  rou = 1.5;
  mv = 1.5;
  iter = 0 ;
  sprank = -ones(max_iter,1);
%   X1 = X0 ;
  tic;
  while iter<=max_iter
    iter = iter + 1;
    X = opt_X(E_omega,Y_omega,W,Z,mv,opt);
    E_omega = opt_E(X,Y_omega,mv,opt);
    W = opt_W(X,Z,mv,opt);
    Y_omega = Y_omega + mv*(E_omega - X .* opt.omega + opt.D_omega); 
    Z = Z + mv*(X - W);
    mv = rou*mv;
    % optimazation condition of Scp ADMM
    [U,sgv,V] = svd(X);
    sgv = diag(sgv) ; 
    idx = sgv>eps(1); Rk = sum(idx);
    RelDist = norm(U(:,idx)'*Gradf(X)*V(:,idx)+...
      lambda*sp*spdiags(sgv(idx).^(sp-1),0,Rk,Rk),'fro')/norm(X,'fro');
% Optimal Condition    
    if exist('ReX','var')
      Rtol = norm(X1-ReX,'fro')/norm(ReX,'fro');
      Rate(iter) = norm(mask.*(X1-ReX),'fro')/norm(mask.*(X0-ReX),'fro');
      spRelErr(iter) = Rtol;
      if Rtol<=tol
        disp('Satisfying the optimality condition:Relative error'); 
        fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
          iter, RelDist, rank(X),Objf(X));
        break;  
      end
    end

    if RelDist<=tol
      disp('Satisfying the optimality condition:Relative Distance'); 
      fprintf('iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X),Objf(X));
      break
    end
    
    if iter==max_iter
      disp("Reach the MAX_ITERATION");
      fprintf( 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
        iter, RelDist, rank(X),Objf(X) );
      break
    end
    sprank(iter) = rank(X);
  end % end while
  estime = toc; 
  
  Par.Xsol = X;  
  Par.time = estime; 
  Par.rank = sprank(1:iter); 
  Par.iterTol = iter ;
end