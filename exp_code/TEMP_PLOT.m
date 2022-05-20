%%
channel = 1;

plot(timeTOTAL.pir{channel},OBJF.pir{channel})
hold on
plot(timeTOTAL.air{channel},OBJF.air{channel})
plot(timeTOTAL.epir{channel},OBJF.epir{channel})
legend("pir","air","epir")



%%