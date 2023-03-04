function weps = update_eps(weps,Rk0,Rk1,rc,nssgv,mu)
% adaptively update strategy

  weps(1:Rk1) = weps(1:Rk1)*mu;
  if Rk0 >= Rk1
    mk0k1 = (weps(Rk0:rc) <= weps(Rk0)); 
    weps(Rk0:rc) = (weps(Rk0:rc).*mk0k1) +  (weps(Rk0).*(~mk0k1)) ;
  else
    mk0k1 = (weps(Rk1:rc) <= weps(Rk1));
    weps(Rk1:rc) = (weps(Rk1:rc).*mk0k1) +  (weps(Rk1).*(~mk0k1)) ;
  end
  if Rk1 < rc && (weps(Rk1+1) > (weps(Rk1)+nssgv ) ) 
    weps(Rk1+1:rc) = mu * (weps(Rk1)+nssgv) .* ones(Rk1+1:rc) ;
  end

end