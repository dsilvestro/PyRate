f="/Users/danielesilvestro/Software/PyRate_github/experimental_code/rootest_q_5_1_epsilon_0.5_fZero_0.25_thr_11.log"
f="/Users/danielesilvestro/Software/PyRate_github/experimental_code/rootest_q_5_1_epsilon_0.5_fZero_0_thr_11.log"


tbl = read.table(f,h=T)
par(mfrow=c(1,3))
plot( tbl$root_true,tbl$root, xlim = c(30,220), ylim = c(30,220),type="n")
points( tbl$root_true,tbl$root_obs, col="red")
abline(0,1,lty=2,col="gray")
segments(x0 =tbl$root_true, x1=tbl$root_true, y0=tbl$root_m2, y1=tbl$root_M2,lwd=4)
segments(x0 =tbl$root_true, x1=tbl$root_true, y0=tbl$root_m4, y1=tbl$root_M4,lwd=2)
segments(x0 =tbl$root_true, x1=tbl$root_true, y0=tbl$root_mth, y1=tbl$root_Mth,lty=1)


#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$root-tbl$root_obs)
#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$mu0_true)
#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$q_med_true)
#plot( (tbl$root-tbl$root_true)/tbl$root_true, tbl$root)


hist( (tbl$root_obs-tbl$root_true)/tbl$root_true)
hist( (tbl$root-tbl$root_true)/tbl$root_true)

hist( (tbl$root_obs-tbl$root_true))
hist( (tbl$root-tbl$root_true))


length( tbl$root_mth[tbl$root_mth > tbl$root_true] )
length( tbl$root_mth[tbl$root_Mth < tbl$root_true] )


mean(abs(tbl$root_obs-tbl$root_true)/tbl$root_true)
mean(abs(tbl$root-tbl$root_true)/tbl$root_true)


hist(tbl$delta_lik)
plot(tbl$delta_lik, tbl$Nobs)
library(HDInterval)
hdi(tbl$delta_lik)
hdi(tbl$delta_lik,0.99)
mean(tbl$delta_lik)
median(tbl$delta_lik)
