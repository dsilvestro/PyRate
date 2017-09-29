# 1 files combined:
# 	C:\Users\Bri\Documents\Career\PyRate\example_files\Canis_example\pyrate_mcmc_logs/Canis_pbdb_data_1_G_marginal_rates.log

# 95% HPDs calculated using code from Biopy (https://www.cs.auckland.ac.nz/~yhel002/biopy/)

pdf(file='C:/Users/Bri/Documents/Career/PyRate/example_files/Canis_example/pyrate_mcmc_logs/Canis_pbdb_data_1_G_marginal_rates_RTT.pdf',width=10.8, height=8.4)
par(mfrow=c(2,2))
library(scales)
L_hpd_m95=c(0.149308120216, 0.149308120216,0.149308120216,0.163606667921,0.163606667921,0.163606667921,0.149308120216,0.149308120216,0.149308120216,0.149308120216)
L_hpd_M95=c(0.403547338478, 0.403547338478,0.403547338478,0.405853970009,0.405853970009,0.405853970009,0.403547338478,0.403547338478,0.403547338478,0.403547338478)
M_hpd_m95=c(0.07895707355, 0.07895707355,0.07895707355,0.07895707355,0.07895707355,0.07895707355,0.07895707355,0.07895707355,0.07895707355,0.07895707355)
M_hpd_M95=c(0.245809461174, 0.245809461174,0.240643883925,0.240643883925,0.240643883925,0.240643883925,0.240643883925,0.240643883925,0.240643883925,0.240643883925)
R_hpd_m95=c(-0.0115362766044, -0.0115362766044,-0.0115362766044,4.939401426e-05,4.939401426e-05,4.939401426e-05,4.939401426e-05,4.939401426e-05,-0.0115362766044,-0.0115362766044)
R_hpd_M95=c(0.291647530738, 0.291647530738,0.291647530738,0.295874488363,0.295874488363,0.295874488363,0.291647530738,0.291647530738,0.2901930986,0.291647530738)
L_mean=c(0.286904270962, 0.286904270962,0.286904270962,0.288443401218,0.290304691424,0.2909934217,0.285981350822,0.286185366345,0.28879061742,0.289314631268)
M_mean=c(0.152839281188, 0.152839281188,0.150544680256,0.150544680256,0.150544680256,0.150544680256,0.150544680256,0.150544680256,0.150544680256,0.150544680256)
R_mean=c(0.134064989774, 0.134064989774,0.136359590706,0.137898720963,0.139760011168,0.140448741444,0.135436670566,0.135640686089,0.138245937164,0.138769951012)
trans=0.5
age=(0:(10-1))* -1
plot(age,age,type = 'n', ylim = c(0, 0.44643936701), xlim = c(-10.5,0.5), ylab = 'Speciation rate', xlab = 'Ma',main='Canis' )
polygon(c(age, rev(age)), c(L_hpd_M95, rev(L_hpd_m95)), col = alpha("#4c4cec",trans), border = NA)
lines(rev(age), rev(L_mean), col = "#4c4cec", lwd=3)
plot(age,age,type = 'n', ylim = c(0, 0.270390407291), xlim = c(-10.5,0.5), ylab = 'Extinction rate', xlab = 'Ma' )
polygon(c(age, rev(age)), c(M_hpd_M95, rev(M_hpd_m95)), col = alpha("#e34a33",trans), border = NA)
lines(rev(age), rev(M_mean), col = "#e34a33", lwd=3)
plot(age,age,type = 'n', ylim = c(-0.0126899042649, 0.325461937199), xlim = c(-10.5,0.5), ylab = 'Net diversification rate', xlab = 'Ma' )
abline(h=0,lty=2,col="darkred")
polygon(c(age, rev(age)), c(R_hpd_M95, rev(R_hpd_m95)), col = alpha("#504A4B",trans), border = NA)
lines(rev(age), rev(R_mean), col = "#504A4B", lwd=3)
plot(age,rev(1/M_mean),type = 'n', xlim = c(-10.5,0.5), ylab = 'Longevity (Myr)', xlab = 'Ma' )
lines(rev(age), rev(1/M_mean), col = "#504A4B", lwd=3)
n <- dev.off()