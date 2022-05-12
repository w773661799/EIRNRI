
% ----------------------------------------------- test IRSVM
format long 
options.Nre = 10 ;
lambda = 1/sqrt(max(size(M)));
tol = 1e-6;
X1 = MC_IRSVM(Xm,Xm,0.5,lambda,M,tol,options) ; 
% -----------------------------------------------