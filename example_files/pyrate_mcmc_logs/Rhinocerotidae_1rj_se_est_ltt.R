
	setwd("/Users/danielesilvestro/Software/PyRate_github/example_files/pyrate_mcmc_logs")
	tbl = read.table(file = "Rhinocerotidae_1rj_se_est_ltt.txt",header = T)
	pdf(file='Rhinocerotidae_1rj_se_est_ltt.pdf',width=12, height=9)
	time = -tbl$time
	library(scales)
	plot(time,tbl$diversity, type="n",ylab= "Number of lineages", xlab="Time (Ma)", main="Range-through diversity through time", ylim=c(0,60),xlim=c(min(time),0))
	polygon(c(time, rev(time)), c(tbl$M_div, rev(tbl$m_div)), col = alpha("#504A4B",0.5), border = NA)
	lines(time,tbl$diversity, type="l",lwd = 2)
	n<-dev.off()
	