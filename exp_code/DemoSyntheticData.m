%% test the accuracy with random data  
% this experiment shows the EPIRNN will convergence faster than PIRNN
% the AIRNN has the similar convergence rate with PIRNN

clc,clear,format long 
rng(22)

nr = 150; nc = 150; 
RankSucc = {};
for r = 10:2:10
% r = 15;
Succ = zeros(1,4);
for irand = 1:1:1

ML = (randn(nr,r)); MR = (randn(nc,r));
Y = MR * MR';
Y = Y/svds(Y,1);

M_org = zeros(nr,nc); 
missrate = 0.1;
for i=1:nc
  randidx = randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i) = 1; 
end
mask = ~M_org; 
Xm = Y.*mask; 
%%
% % % % parameters for Schatten-p norm Regularization model
  lambda = 1e-4*norm(Xm,inf);
  itmax = 1e3; 
  sp = 0.9; 
  tol = 1e-9; 
%   X0 = zeros(size(Y));
  X0 = randn(size(Y));
  Lip = 1.1;
  adaeps = 1e0;
%%
% % % generate the IRNN_Lu's parameters ------
%   omega = find(mask);
%   [I,J] = ind2sub([nr,nc], omega);
%   col = [0; find(diff(J)); length(omega)];
%   V = UVtOmega(ML, MR, I, J, col);
%   D = spconvert([I,J,V; nr, nc, 0]);
%   M = opRestriction(nr*nc, omega);
%   vecx = Y(:);
%   pvecx = M(vecx,1);
%   fun_Lu = 'lp' ;
% 
%   optionsLu.max_iter = itmax;
%   optionsLu.gamma = sp;
%   optionsLu.lambda_Init = lambda;
%   optionsLu.lambda_rho = 0.9; 
%   optionsLu.lambda_Target = lambda;
%   optionsLu.tol = tol;
%   optionsLu.mu = Lip;
%   SolLu = IRNN_Lu(X0, fun_Lu, pvecx, M, nr, nc, optionsLu);
%% 
% % % generate the PIRNN's parameters ------ 
  optionsP.max_iter = itmax; 
  optionsP.eps = 1e-2;  
  optionsP.beta = Lip; 
  optionsP.KLopt = tol;
  Solpir = MC_PIRNN(X0,Xm,sp, lambda, mask, tol, optionsP); 
% 
%% 
% % % % generate the AIRNN's parameters ------ 
  optionsA.max_iter = itmax;   
  optionsA.eps = adaeps;
  optionsA.beta = Lip; 
  optionsA.KLopt = tol;
  optionsA.mu = 0.1;
  Soladair = MC_AIRNN(X0,Xm,sp, lambda, mask, tol, optionsA); 
%%
% % % generate the EPIRNN's parameters ------ 
  optionsEP.max_iter = itmax;   
  optionsEP.eps = adaeps;
  optionsEP.beta = Lip; 
  optionsEP.KLopt = tol;
  optionsEP.mu = 0.1;
  optionsEP.alpha = 5e-1; 
  Solepir = MC_EPIRNN(X0,Xm,sp, lambda, mask, tol, optionsEP);
%%
% Succ(irand,1) = norm(Y - SolLu.Xsol,'fro')/norm(Y,'fro');
% Succ(irand,2) =  norm(Y - Solpir.Xsol,'fro')/norm(Y,'fro');
% Succ(irand,3) =  norm(Y - Soladair.Xsol,'fro')/norm(Y,'fro');
% Succ(irand,4) =  norm(Y - Solepir.Xsol,'fro')/norm(Y,'fro');
end % end for irand 100 
% RankSucc{end+1} = Succ;
end % end for rank iteration
%% 
% % % plot the successful recovery

%% ---------- plot the rank and objective
plot(Solpir.rank);hold on
plot(Soladair.rank)
plot(Solepir.rank); hold off
legend("PIR","AIR","EPIR")
%%
relative_err = norm(Y-Soladair.Xsol,'fro')/norm(Y,'fro')
%%
Tomega = sub2ind([I,J],m,n)