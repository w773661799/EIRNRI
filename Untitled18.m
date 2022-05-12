PIR = spPIR{1};
AIR = spAIR{1};

%
PIRx = (1:1:PIR.iterTol)/1e2; AIRx = (1:1:AIR.iterTol)/1e2; 
% ------------ RelErr plot 
figure(1)
plot(PIRx,log10(PIR.RelErr),':.k','linewidth',1);hold on
plot(AIRx,log10(AIR.RelErr),'-.r','linewidth',1);hold off
title("RelErr "); 
set(gca,'yscale','log')
xlabel("iteration"); ylabel("RelErr")
legend("PIRNN","AIRNN")

% ------------ RelDist plot 
figure(2)
plot(PIRx,log10(PIR.RelDist),':.k','linewidth',1);hold on
plot(AIRx,log10(AIR.RelDist),'-.r','linewidth',1);hold off
title("RelDist "); 
set(gca,'xscale','log')
xlabel("iteration"); ylabel("RelDist")
legend("PIRNN","AIRNN")
% % ------------ obj plot 
figure(3)
plot(PIRx,PIR.f,':.k','linewidth',1);hold on; 
plot(AIRx,AIR.f,'-.r','linewidth',1);hold off
title("Objective"); 
% set(gca,'xscale','log')
xlabel("iteration"); ylabel("F(x)")
legend("PIRNN","AIRNN")
% % plot(px,Fp,'-+r','linewidth',1);hold on
% % plot(px,Fa,'--sb','linewidth',1);hold on
% % plot(px,Fe,'-.g','linewidth',1);hold on
% % legend("PIRNN","AIRNN","EPIRNN")
% plot(px,Fp,'-r');hold on; plot(px,Fa,'--b');
% plot(px,Fe,'-.g');hold off
% legend("PIRNN","AIRNN","EPIRNN")

% figure(3)

% ------------ rank plot
figure(4)
plot(PIRx,PIR.rank,':.k','linewidth',1);hold on; 
plot(AIRx,AIR.rank,'-.r','linewidth',1);hold off
title("rank of iterations")
% set(gca,'xscale','log')
xlabel("iteration"); ylabel("rank")
legend("PIRNN","AIRNN")
% ------------  time plot
