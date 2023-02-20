%% ---------- Robust Experiment ----------
% Rank Robust
% rank from 5 to 10, record the recognized success rank
clc,clear,format long; rng(22)
nr = 150; nc = 150; r = 10 ;
% --------------- Synthetic data ---------------
Y = randn(nr,r) * randn(r,nc);
% --------------- random mask ---------------
M_org = zeros(nr,nc); missrate = 0.5; 
for i=1:nc 
  randidx=randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~M_org; Xm = Y.*mask;


% %% --------------- parameters ---------------
lambda = 1e-1 * norm(Xm,inf);
itmax = 5e3; 
sp = 0.5; 
tol = 1e-7; 
klopt = 1e-5;
weps = 1e-16;

options.Rel = Y; 
options.max_iter = itmax; 
options.KLopt = klopt;
options.beta = 1.1; 
options.teps = weps;


Rank = [5,10];
WEPS = [1e-1,5e-2,1e-2];
times = 100;
init_rank_max = 20;

Rank_RC.PIR = zeros(length(Rank),length(WEPS));
Rank_RC.AIR = zeros(size(Rank));
Rank_RC.EPIR = zeros(size(Rank));

% Initial point
for irank = 1:length(Rank) %3
  r = Rank(irank);
  parfor r_iter = 1:init_rank_max %  
    par{r_iter} = VsRobustEps(nr,nc,r,r_iter,lambda,sp,missrate,tol,options,WEPS ,0,times);
% -------------------------------------
  end
  for iter = 1:init_rank_max
    Rank_RC.PIR(irank,:) = Rank_RC.PIR(irank,:) + par{iter}.PIR;
    Rank_RC.AIR(irank) = Rank_RC.AIR(irank) + par{iter}.AIR;
    Rank_RC.EPIR(irank) = Rank_RC.EPIR(irank) + par{iter}.EPIR;
  end
end
%%
% save(Robust_Eps.mat,Robust,'-mat')
save("..\exp_cache\Rank_Robust.mat","Rank_RC",'-mat')
%%
irankplt = 2;
X = categorical({'EPIRNN','AdaIRNN', '\epsilon=10^{-1}', '\epsilon=10^{-2}','\epsilon=10^{-3}' });
% X = reordercats(X,{'Medium','Extra Large'});
X_num = [1,2,4,5,6];
bar(X,[Rank_RC.EPIR(irankplt), Rank_RC.AIR(irankplt) , Rank_RC.PIR(irankplt,:)])
ylabel('# Number of success')
% xlabel(['EPIRNN',1,3,4,5])
%%
% for irank = 1:length(Rank) %3
%   r = Rank(irank);
%   parfor r_iter = 1:init_rank_max % 74 
%     par{r_iter} = VsRobustEps(nr,nc,r,r_iter,lambda,sp,missrate,tol,options,REPS ,0,1);
% % -------------------------------------
%   end
%   for iter = 1:init_rank_max
%     Robust.PIR(irank,:) = Robust.PIR(irank,:) + par{iter}.PIR;
%     Robust.AIR(irank) = Robust.AIR(irank) + par{iter}.AIR;
%     Robust.EPIR(irank) = Robust.EPIR(irank) + par{iter}.EPIR;
%   end
% end
% 

%% ----------  percentage of success ----------
% AdaIRNN V.S. ProxIRNN with random data 
% this experiment shows the AdaIRNN are more robust than ProxIRNN
% the AdaIRNN has the similar convergence rate with ProxIRNN if the 
% initialization eps are similar
% --------------- generate data ---------------
clc,clear,format long; rng(23); 
% p = parpool(16); 
nr = 150; nc = 150; 
% %% --------------- parameters ---------------
lambda = 5e-2;
itmax = 5e3;
sp = 0.5;
tol = 1e-7;
klopt = 1e-5;
beta = 1.1;

success = 1e-2;
% ------------------------------------------------------------------------
missrate = 0.5; 
weps = 1e-4;

times = 10;
init_rank_max = 20;

WEPS = [1e-2, 1e-3, 5e-4, 1e-4];
options.max_iter = itmax;
options.KLopt = klopt;
%   options.eps = weps;
options.beta = beta;

% -------------------------- 75 *20 times --------------------------
% with different initialization rank: 0--74
% for each initialization rank we test 20 times
Rank = [5,10,15,20,25,30];

Robust.PIR = zeros(length(Rank),length(WEPS));
Robust.AIR = zeros(size(Rank));
Robust.EPIR = zeros(size(Rank));

options.max_iter = itmax;
options.KLopt = klopt;
%       options.eps = weps;
options.beta = beta;

for irank = 1:length(Rank) %3
  r = Rank(irank);
  parfor r_iter = 1:init_rank_max % 74 
      r_iter
    par{r_iter} = VsRobustEps(nr,nc,r,r_iter,lambda,sp,missrate,tol,options,WEPS,[1,success],1);
% -------------------------------------
  end
  for iter = 1:init_rank_max
    Robust.PIR(irank,:) = Robust.PIR(irank,:) + par{iter}.PIR;
    Robust.AIR(irank) = Robust.AIR(irank) + par{iter}.AIR;
    Robust.EPIR(irank) = Robust.EPIR(irank) + par{iter}.EPIR;
  end
end

%%
% save(Robust_Eps.mat,Robust,'-mat')
save("..\exp_cache\Robust_Eps_mu95_r535_0217done.mat","Robust",'-mat')
%%
delete(p);