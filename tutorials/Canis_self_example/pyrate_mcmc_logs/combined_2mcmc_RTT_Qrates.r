library(scales)


pdf(file="C:/Users/SimoesLabAdmin/Documents/PyRate/tutorials/Canis_self_example/pyrate_mcmc_logs/combined_2mcmc_RTT_Qrates.pdf",width=0.6*9, height=0.6*7)

age = c(-21.124025, -5.333)
Q_mean = 0.20576091048638478
Q_hpd_m = 0.061357
Q_hpd_M = 0.407408
plot(age,age,type = 'n', ylim = c(0, 116.242225), xlim = c(-21.124025,-0.0), ylab = 'Preservation rate', xlab = 'Ma',main='Preservation rates' )
segments(x0=age[1], y0 = Q_mean, x1 = age[2], y1 = Q_mean, col = "#756bb1", lwd=3)
polygon( c(age, rev(age)), c(Q_hpd_m, Q_hpd_m, Q_hpd_M, Q_hpd_M), col = alpha("#756bb1",0.5), border = NA)
age = c(-5.333, -2.58)
Q_mean = 3.4435294446862317
Q_hpd_m = 2.231234
Q_hpd_M = 4.679135
segments(x0=age[1], y0 = 0.20576091048638478, x1 = age[1], y1 = Q_mean, col = "#756bb1", lwd=3)
segments(x0=age[1], y0 = Q_mean, x1 = age[2], y1 = Q_mean, col = "#756bb1", lwd=3)
polygon( c(age, rev(age)), c(Q_hpd_m, Q_hpd_m, Q_hpd_M, Q_hpd_M), col = alpha("#756bb1",0.5), border = NA)
age = c(-2.58, -0.0117)
Q_mean = 11.443939723048873
Q_hpd_m = 10.285854
Q_hpd_M = 12.608629
segments(x0=age[1], y0 = 3.4435294446862317, x1 = age[1], y1 = Q_mean, col = "#756bb1", lwd=3)
segments(x0=age[1], y0 = Q_mean, x1 = age[2], y1 = Q_mean, col = "#756bb1", lwd=3)
polygon( c(age, rev(age)), c(Q_hpd_m, Q_hpd_m, Q_hpd_M, Q_hpd_M), col = alpha("#756bb1",0.5), border = NA)
age = c(-0.0117, -0.0)
Q_mean = 81.16355241351526
Q_hpd_m = 65.920461
Q_hpd_M = 97.586888
segments(x0=age[1], y0 = 11.443939723048873, x1 = age[1], y1 = Q_mean, col = "#756bb1", lwd=3)
segments(x0=age[1], y0 = Q_mean, x1 = age[2], y1 = Q_mean, col = "#756bb1", lwd=3)
polygon( c(age, rev(age)), c(Q_hpd_m, Q_hpd_m, Q_hpd_M, Q_hpd_M), col = alpha("#756bb1",0.5), border = NA)
n <- dev.off()