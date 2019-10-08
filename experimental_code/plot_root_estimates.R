f="/Users/danielesilvestro/Software/PyRate_github/experimental_code/rootest_q_5_1_epsilon_0.5_fZero_0.25_thr_11.log"
f="/Users/danielesilvestro/Dropbox (Personal)/AngiospermOrigin/fossil_data/rootest_q_8.52_1_epsilon_0.5_fZero_0.1_thr_11.log"
f="/Users/danielesilvestro/Dropbox (Personal)/AngiospermOrigin/fossil_data/rootest_q_8.52_1_epsilon_0.5_fZero_0.1_thr_11_incr_q.log"

library(scales)
library(HDInterval)
tbl = read.table(f,h=T)
par(mfrow=c(1,3))
indxWFOSS = which(tbl$Nfossils>0)
tbl = tbl[indxWFOSS,]
plot( tbl$root_true,tbl$root_est, xlim = c(0,200), ylim = c(0,200),pch = 19,col=alpha("black",0.2),type="n")
points( tbl$root_true,tbl$root_obs, col=alpha("red",0.2),pch = 19)
abline(0,1,lty=2,col="gray")

segments(x0 =tbl$root_true, x1=tbl$root_true, y0=tbl$root_m2, y1=tbl$root_M2,lwd=4,col=alpha("black",0.2))
segments(x0 =tbl$root_true, x1=tbl$root_true, y0=tbl$root_m4, y1=tbl$root_M4,lwd=2,col=alpha("black",0.2))
segments(x0 =tbl$root_true, x1=tbl$root_true, y0=tbl$root_mth, y1=tbl$root_Mth,lty=1,col=alpha("black",0.2))


#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$root-tbl$root_obs)
#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$mu0_true)
#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$q_med_true)
#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$root)


hist( (tbl$root_obs-tbl$root_true)/tbl$root_true)
hist( (tbl$root_est-tbl$root_true)/tbl$root_true)

hist( (tbl$root_obs-tbl$root_true))
hist( (tbl$root_est-tbl$root_true))
hist( (tbl$root_est-tbl$root_obs))


length( tbl$root_mth[tbl$root_mth > tbl$root_true] )
length( tbl$root_mth[tbl$root_Mth < tbl$root_true] )
length( tbl$root_m4[tbl$root_m4 > tbl$root_true] )/length( tbl$root_m4)
length( tbl$root_m4[tbl$root_M4 < tbl$root_true] )/length( tbl$root_m4)

which(tbl$root_M4 < tbl$root_true)



mean(abs(tbl$root_obs-tbl$root_true)/tbl$root_true)
mean(abs(tbl$root_est-tbl$root_true)/tbl$root_true)


hist(tbl$delta_lik)
summary(tbl$delta_lik)

plot(tbl$delta_lik, tbl$Nobs)
library(HDInterval)
hdi(tbl$delta_lik)
hdi(tbl$delta_lik,0.99)
mean(tbl$delta_lik)
median(tbl$delta_lik)
