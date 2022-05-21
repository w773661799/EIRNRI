%% without Warm START
% START with obsevied points
% the parameters as:
% lambda = norm(M,"fro")*1e-3, sp = 0.25
% the other parameters are the default value

%% matrix completion form the random mask
clear;clc;format long 
start = 1;

while start==1
start=0;

rng(22);format long
cwd = fileparts(pwd) ;
path_lena = strcat(cwd,'\img_image\lena.png');
path_re1 = strcat(cwd,'\img_image\re1.jpg');
img_ori = double(imread(path_lena))/255 ; 
% img_ori = double(imread(path_lena))/255;
img_size = size(img_ori);

%% mask
  %% random mask
  SR = 0.5; % sampleRate = 1 - missRate
  mask = zeros(img_size(1:2));
  for i=1:img_size(2)
      idx = 1:1:img_size(1) ;
      randidx = randperm(img_size(1),img_size(1)); % 随机[n] 中的 k 个 index
      mask(randidx(1:ceil(img_size(1)*SR)),i)=1; 
  end
  % mask should obtain (1-missrate)*m*n elements from the original image
%% strictly low rank
% save the top 20% of the singular value
  rt = ceil(min(size(img_ori(:,:,1)))/5); 
  for i=1:3
    [U,S,V]=svd(img_ori(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
  end

%% ------------------------ RECOVERY ------------------------

XM = mask.*Xt;
tol = 1e-5 ;
mu = 1.3; 
maxIter = 2e3;
options.max_iter = maxIter; 
options.mu = mu;  
options.KLopt = 1e-5;
sp = 0.1; 

%% PIRNN
optionsP = options; optionsP.eps = 1e-15;
for channel=1:3
  Xm = XM(:,:,channel); 
  lambda = norm(Xm,"fro")*1e-2;
  PIR = MC_PIRNN(Xm,Xm,sp, lambda, mask, tol, optionsP);
  imgR.pir(:,:,channel) = PIR.Xsol;
  timeTotal.pir{channel} = PIR.time;
  Objective.pir{channel} = PIR.f;
  iterRank.pir{channel} = PIR.rank;
end
disp("---------------------------------- PIRNN")
%% AIRNN
optionsA = options; optionsA.eps=1e1;
optionsA.Scalar = 0.1; 
for channel=1:3
  Xm = XM(:,:,channel); 
  lambda = norm(Xm,"fro")*1e-2;
  AIR = MC_AIRNN(Xm,Xm,sp, lambda, mask, tol, optionsA);
  imgR.air(:,:,channel) = AIR.Xsol;
  timeTotal.air{channel} = AIR.time;
  Objective.air{channel} = AIR.f;
  iterRank.air{channel} = AIR.rank;
end
disp("---------------------------------- AIRNN")
%% EPIRNN
optionsEP = options; optionsEP.eps=1e1;
optionsEP.Scalar = 0.1; optionsEP.alpha = 0.75; 
for channel=1:3
  Xm = XM(:,:,channel); 
  lambda = norm(Xm,"fro")*1e-2;
  EPIR = MC_EPIRNN(Xm,Xm,sp, lambda, mask, tol, optionsEP);
  imgR.epir(:,:,channel) = EPIR.Xsol;
  timeTotal.epir{channel} = EPIR.time;
  Objective.epir{channel} = EPIR.f;
  iterRank.epir{channel} = EPIR.rank;
end
disp("---------------------------------- EPIRNN")
%% SCP ADMM
optionsSCP.max_iter = 5e2;
optionsSCP.tau = 30;
for channel=1:3
  Xm = XM(:,:,channel); 
  lambda = norm(Xm,"fro")*1e-2;
  SCP = MC_SCpADMM(Xm, sp, lambda, mask, tol, optionsSCP);
  imgR.scp(:,:,channel) = SCP.Xsol;
  timeTotal.scp{channel} = SCP.time;
  Objective.scp{channel} = SCP.f;
  iterRank.scp{channel} = SCP.rank;
end
disp("---------------------------------- SCPADMM")
%% IRNN
fun_irnn = 'lp'; 
optionsIRNN.max_iter = 2e3;
optionsIRNN.gamma = sp;
optionsIRNN.tol = tol;
optionsIRNN.mu = mu;
m = img_size(1); n = img_size(2);
M_IRNN = opRestriction(m*n,find(mask==1));

for channel=1:3
  Xm = XM(:,:,channel); 
  xIR = Xm(:);
  y = M_IRNN(xIR,1);
  optionsIRNN.lambda_Init = norm(Xm)*1e2;
  optionsIRNN.lambda_Target = norm(Xm,"fro")*1e-1;
  optionsIRNN.lambda_rho = 0.65; 
  Sol_IRNN = IRNN(xIR, fun_irnn, y, M_IRNN, m, n, optionsIRNN);
  imgR.irnn(:,:,channel) = Sol_IRNN.Xsol;
end
disp("---------------------------------- IRNN")

%% FGSR
optionsFGSR.tol=1e-5;
optionsFGSR.maxiter = maxIter;
optionsFGSR.p = sp;
for channel=1:3
  Xm = XM(:,:,channel); 
  optionsFGSR.lambda = norm(Xm,"fro")*1e-2;
  [X,~,~] = MC_FGSRp_PALM(Xm,Xm,optionsFGSR);
end
disp("---------------------------------- IRNN")


end
%% 








