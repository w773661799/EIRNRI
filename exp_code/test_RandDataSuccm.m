%% the Accuracy of recovery within PIRNN / AIRNN / EPIRNN for random data  
% this experiment shows the EPIRNN will convergence faster than PIRNN
% the AIRNN has the similar convergence rate with PIRNN
% AdaIRNN and EPIRNN has better performence 


clear; clc; format long
rng(22)

% --------------------- Synthetic data with low rank ---------------------
nr = 150; nc = 150; lrk = 15;
Y = randn(nr,nc);
[uY,sY,vY] = svd(Y);
Y = uY* diag([svds(sY,lrk);zeros(nr-r,1)])*vY';
Y = Y(:,randperm(nc)); 
clear uY sY vY

% ------------------------------ random mask ------------------------------
M_org = zeros(nr,nc); 
missrate = 0.5; 
for i=1:nc 
  idx = 1:1:nr;
  randidx = randperm(nr,nr); % random sequence
  M_org(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~M_org; Xm=Y.*mask; 

% ------------- parameters for Schatten-p norm regularization -------------
lambda = 1e-3*norm(Xm,"fro");
itmax = 5e3; 
sp = 0.1; 
tol = 1e-5; 
X0 = zeros(size(Y));