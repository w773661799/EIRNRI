% clear;clc


%% % 
img_show = Tab_img{1};
%%
figure(1)
imshow(img_show.ori_img)

figure(2)
imshow(img_show.mask_img)

figure(3)
imshow(img_show.low_img)

%% plot lena_best_lambda
format bank
R_psnrTable = [];

for irank = 1:6
img_show = Tab_img{irank};

for i = 1:5
    tempR = 0;
    img_sol = img_show.sol{i};
    for k =1 :3
      tempR = tempR + rank(img_sol(:,:,k));
    end
    R_psnrTable(irank,2*i-1:2*i) = [psnr(img_sol,img_show.ori_img),floor(tempR/3)];    
    
    figure(i)
    imshow(img_sol)
%   imshow(img_sol(:,:,1))
end


end
%%
for i = 1:6
  img_show = Tab_img{i};
  img_ori = img_show.ori_img;
  img_low = img_show.low_img;
  for j = 1:5
    img_sol = img_show.sol{j};
%     sPsnr = psnr(img_ori,img_sol)
    R_psnr_Rank_Table(i,2*j-1) = vpa( psnr(img_ori,img_sol), 2);
    tempR = 0;
    for k =1 :3
      tempR = tempR + rank(img_sol(:,:,k));
    end
    R_psnr_Rank_Table(i,2*j) = floor(tempR/3);
  end
end

%% with ch1 red

for i = 1:5
    tempR = 0;
    img_sol = img_show.sol{i};
    for k =1 :3
      tempR = tempR + rank(img_sol(:,:,k));
    end
    [psnr(img_sol,img_show.ori_img),floor(tempR/3)];
    figure(i)
    
%   imshow(img_show.sol{i}.*ch1)
  
  
  ich = 1;
  for i=1:3
    if i ~= ich
      img_sol(:,:,i) = img_sol(:,:,i).*ch1; 
    else
      img_sol(:,:,i) = img_sol(:,:,i).*ch1 + (1-ch1); 
    end
  end
  imshow(img_sol)
%   imshow(img_sol(:,:,1))


end



%%

ch1 = img_sol(:,:,1);
ch1 = ones(size(img_sol(:,:,1)));
bd = 1;
l = 190;
r = 250;
s = 240;
x = 290;

ch1(s:x,l:l+bd) = 0;
ch1(s:x,r:r+bd) = 0;
ch1(s:s+bd,l:r) = 0;
ch1(x:x+bd,l:r+bd) = 0;
imshow(ch1)
%%
line_M = zeros(size(img_sol(:,:,1)));
line_M(270:270:280) = 0.5; 
imshow(line_M)
%%
for i = 1:5
    tempR = 0;
    for k =1 :3
      tempR = tempR + rank(img_sol(:,:,k));
    end
    [psnr(img_sol,img_show.ori_img),floor(tempR/3)];
  figure(i)
  img_sol = img_show.sol{i};
%   imshow(img_show.sol{i}.*ch1)
  
  
  ich = 1;
  for i=1:3
    if i ~= ich
      img_sol(:,:,i) = img_sol(:,:,i).*ch1; 
    else
      img_sol(:,:,i) = img_sol(:,:,i).*ch1 + (1-ch1); 
    end
  end
  imshow(img_sol)
%   imshow(img_sol(:,:,1))


end
%% for table

% clear;clc
% R_psnr_Rank_Table = zeros(6,10);
format bank
for i = 1:6
  img_show = Tab_img{i};
  img_ori = img_show.ori_img;
  img_low = img_show.low_img;
  for j = 1:5
    img_sol = img_show.sol{j};
%     sPsnr = psnr(img_ori,img_sol)
    R_psnr_Rank_Table(i,2*j-1) = vpa( psnr(img_ori,img_sol), 2);
    tempR = 0;
    for k =1 :3
      tempR = tempR + rank(img_sol(:,:,k));
    end
    R_psnr_Rank_Table(i,2*j) = floor(tempR/3);
  end
end

%%
digits(2)
for i = 1:6
  for j =1:10
    Temp_table(i,j)  = vpa(R_psnr_Rank_Table(i,j),2);

  end
end
