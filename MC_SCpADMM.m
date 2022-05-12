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
  iter = 0 ;goon = true;
  sprank = -ones(max_iter,1);
%   X1 = X0 ;
  tic;
  while iter<=max_iter && goon
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
    if exist('ReX','var')
      Rtol = norm(X-ReX,'fro')/norm(ReX,'fro');
      goon = (Rtol>tol)&&(RelDist>tol); 
    else
      goon = RelDist>tol;
    end
    sprank(iter) = rank(X);
% what if the two iterations is colse? 
% there is no evidence shows that we can terminate iteration if the two is so close
%     goon = goon && (norm(X-X1,"fro")+(1-Scalar)*norm(weps(1:Rk),1)/Scalar > KLopt);
%     X1 =X; 
% -----
    if (iter == 1)||(mod(iter,5e8) == 0)||(~goon)||(iter==max_iter)
      fprintf(1, 'iter:%04d\t err:%06f\t rank(X):%d\t Obj(F):%d\n', ...
              iter, RelDist, rank(X),Objf(X) );
            %nnz(X1(~unobserved))
    end
  end % end while
  estime = toc; 
  
  Par.Xsol = X;  
  Par.time = estime; 
  Par.rank = sprank(1:iter); 
  Par.iterTol = iter ;
end