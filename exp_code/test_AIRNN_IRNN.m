X0 = xIR; fun = fun_irnn ; 
M= M_IRNN;
options = optionsIRNN ; 

  if isfield(options,"max_iter"), max_iter = options.max_iter; else, max_iter = 5e3;end
  if isfield(options,"gamma"), gamma = options.gamma ;  else, gamma = 1; end
  if isfield(options,"lambda_Init"), lambda = options.lambda_Init; else, lambda = max(abs(M(y,2)));end
  if isfield(options,"lambda_rho"), lambda_rho = options.lambda_rho; else, lambda_rho = 0.9; end
  if isfield(options,"lambda_Target"), lambda_Target = options.lambda_Target;else, lambda_Target = max(abs(M(y,2)))*1e-5; end
  if isfield(options,"tol"), tol = options.tol ; else, tol = 1e-5; end
  if isfield(options,"mu"), mu = options.mu ;  else, mu = 1.1;end
  Objf = @(x,X)(norm(y-M(x,1),2)^2/2 + lambda_Target*norm(svds(X,rank(X)),gamma)^(gamma));

  hfun_sg = str2func([fun '_sg']);
  insweep = 2e2; % warm  start step
  x = zeros(m*n,1);
  X = reshape(x,[m,n]);
  iter = 0;
  f_current = norm(y-M(x,1)) + lambda*norm(x,1);
  while lambda > lambda_Target
    ftol = @(x)(norm(y-M(x,1)) + lambda*norm(x,1));
    f_current = ftol(x) ; 
    for ins = 1 : insweep    
      iter = iter + 1;
         if iter == 5
         error("iter")
         end
% save for plot and table       
      Stime(iter) = toc;
%       sprank(iter) = rank(X);
%       spf(iter) = Objf(x,X);

      f_previous = f_current;
      x = x + (1/mu)*M(y - M(x,1),2);
      X_md(:,iter) =x;
      [U,S,V] = svd(reshape(x,[m,n]),'econ');
      sigma = diag(S);
      w = hfun_sg(sigma,gamma,lambda);
            fprintf('md %d\t %d\t %d\n ',norm(x),norm(w),iter)
      sigma = sigma - w/mu;
      svp = length(find(sigma>0));
      sigma = sigma(1:svp);
      X = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
      x = X(:);
      f_current = norm(y-M(x,1)) + lambda*norm(x,1);
%       f_current = ftol(x);
      if norm(f_current-f_previous)/norm(f_current + f_previous) < tol
        disp("first brek ----------------------------------------------")
        break;
      end
    end
    if norm(y-M(x,1)) < tol
      break
    end
    lambda = lambda*lambda_rho; 
  end

%%
lambda_Init= optionsIRNN.lambda_Init;

iter = 0 ;
hfun_sg = str2func([fun '_sg']);
lambda = lambda_Init;
lambda_Target = lambda_Init * 1e-5;
mu = 1.1;
insweep = 200;
x = zeros(m*n,1);
f_current = norm(y-M(x,1)) + lambda*norm(x,1);
while lambda > lambda_Target
     for ins = 1 : insweep    
         iter = iter +1;
         if iter == 5
         error("iter")
         end
          
         f_previous = f_current;
         x = x + (1/mu)*M(y - M(x,1),2);
         X_lu(:,iter) = x;
         [U,S,V] = svd(reshape(x,[m,n]),'econ');
         sigma = diag(S);
         w = hfun_sg(sigma,gamma,lambda);
               fprintf('lu %d\t %d\t %d \n ',norm(x),norm(w),iter)
         sigma = sigma - w/mu;
         svp = length(find(sigma>0));
         sigma = sigma(1:svp);
         X = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
         x = X(:);
         f_current = norm(y-M(x,1)) + lambda*norm(x,1);
         if norm(f_current-f_previous)/norm(f_current + f_previous) < tol
           disp("first brek ----------------------------------------------")
             break;
         end
     end
    if norm(y-M(x,1)) < tol
      disp("Satisfying the optimal condition");
%       fprintf( 'iter:%04d\t  rank(X):%d\t Obj(F):%d\n', ...
        break;
    end
    lambda = lambda_rho*lambda;
end
