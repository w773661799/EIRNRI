clear; clc;rng(22);format long 
% addpath('tools')
% addpath('data');
% dataset = {'movielens1m'};
dataset = {'lena'};

X_img = imread('sistlibrary.jpeg');
% X=imread('im2.jpg');
% X=imresize(X,0.5);% im1 0.5
X = double(X_img)/255;
% [U,S,V]=svd(X);
% figure(1), imshow(X)

% rt = ceil(size(X(:,:,1),2)/3);
rt = ceil(min(size(X(:,:,1)))/5); 
% SVD : U*S*V' = svd(X)
%       VecSigma = svd(X)

%

for i=1:3
  [U,S,V]=svd(X(:,:,i));
  Sigma(:,:,i) = S ; 
  Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
end

[nr,nc] = size(X(:,:,1));
missrate = 0.5;
mask = zeros(nr,nc);
for i=1:nc  
    idx = 1:1:nr;
    randidx=randperm(nr,nr); % 随机[n] 中的 k 个 index
    mask(randidx(1:ceil(nr*missrate)),i)=1; 
end
mask = ~mask;

XM = mask.*Xt;
data = XM(:,:,1);
%%
for chn = 1:3
%     load(['data/',dataset{j},'.mat']);
    %%% arr = data;
    data = Xt(:,:,chn); 
    para.data = XM(:,:,chn); 
    [m, n] = size(data);

    para.test.data = X(:,:,chn);
    para.test.m = m;
    para.test.n = n;
    
    [row, col, val] = find(data);
    
    val = val - mean(val);
    val = val/std(val);
    idx = randperm(length(val));

    traIdx = idx(1:floor(length(val)*0.5));
    tstIdx = idx(ceil(length(val)*0.5): end);

    clear idx;

    traData = sparse(row(traIdx), col(traIdx), val(traIdx));
    traData(size(data,1), size(data,2)) = 0;

    para.test.row  = row(tstIdx);
    para.test.col  = col(tstIdx);
    para.test.data = val(tstIdx);
%     clear data;
%%
%     lambda = 1e-3 * norm(data,inf);
    lambda = 0;
    theta = 0.5;

    para.maxR = 55;
    para.maxtime = 200;

    para.regType = 4;
    para.maxIter = 1e4;
    para.tol = 1e-9;
    R = randn(n, para.maxR); % 
    para.R = R;

    U0 = powerMethod(traData, R, para.maxR, 1e-6);    
    para.U0 = U0;

%% IRNN

    [~, ~, ~, out] = AccIRNN( traData, lambda, 0.5, para );

%  
XAcc_lsvd(:,:,chn) = out.Q * out.U * out.V';
end
%     plot(out{2}.Time, log(out{2}.obj), 'r');
%     hold on;
% 
%      legend('IRNN', 'AIRNN');
imshow(XAcc_lsvd)
    
