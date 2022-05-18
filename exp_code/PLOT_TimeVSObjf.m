for plotix =1:1
  plot(timeTOTAL.air{plotidx}(100:end),OBJF.air{plotidx}(100:end))
  hold on
  plot(timeTOTAL.epir{plotidx}(100:end),OBJF.epir{plotidx}(100:end))
end
legend("AIR","EPIR")

%%
plot(PIR.time,PIR.f);hold on
plot(AIR.time,AIR.f)
plot(EPIR.time,EPIR.f);hold off
legend("PIR","AIR","EPIR")