plot_DE_rates <- function(f,A1,A2,n_lines=100,binsize=0.5,Alpha = 0.5,YlimD = 0.25,YlimE = 0.5,random_sample=F){
	library(scales)
	t = read.table(f,h=T)
	cn  = colnames(t)
	ind = grep("d12",cn)

	times = c()
	for (i in ind){
		times = c(times,strsplit(cn[i],"_")[[1]][2])
	}
	times = -as.numeric(times)*binsize

	plot_vaues= (dim(t)[1]-n_lines):dim(t)[1]
	
	if (random_sample){
		# burnin = 0.5
		plot_vaues= sample((0.5*dim(t)[1]):dim(t)[1],size=n_lines)
		(dim(t)[1]-n_lines):dim(t)[1]	
	}

	add_plot <- function(ind,color){
		for (i in plot_vaues){
			v= (as.numeric(as.vector(t[i,ind])))
			lines(times,v,col=alpha(color,Alpha),lwd=3)
		}
	}

	par(mfrow=c(2,4))
	ind = grep("d12",cn)
	plot(times,as.numeric(as.vector(t[i,ind])),type="n",main=paste("Dispersal",A1,"->",A2),
		ylim=c(0,YlimD),ylab="Dispersal rate",xlab="Time")
	add_plot(ind,"#4c4cec")

	ind = grep("d21",cn)
	plot(times,as.numeric(as.vector(t[i,ind])),type="n",main=paste("Dispersal",A2,"->",A1),
		ylim=c(0,YlimD),ylab="Dispersal rate",xlab="Time")
	add_plot(ind,"#4c4cec") #"#4c4cec"
	
	f1 = gsub("_marginal_rates.log",".log",f)
	#f1 = gsub("marginal_rates/","",f1)
	mcmc_t = read.table(f1,h=T)
	mcmc_t = mcmc_t[plot_vaues,]
	
	boxplot(mcmc_t[grep("cov_d",colnames(mcmc_t))]) #, ylim = c(min(mcmc_t[grep("cov_d",colnames(mcmc_t))])*0.7,1.3*max(mcmc_t[grep("cov_d",colnames(mcmc_t))])))
	boxplot(mcmc_t[grep("q1",colnames(mcmc_t))]   ) #, ylim = c(min(mcmc_t[grep("q1",colnames(mcmc_t))]   )*0.7,1.3*max(mcmc_t[grep("q1",colnames(mcmc_t))]   )))

	ind = grep("e1",cn)
	plot(times,as.numeric(as.vector(t[i,ind])),type="n",main=paste("Extinction",A1),
		ylim=c(0,YlimE),ylab="Extinction rate",xlab="Time")
	add_plot(ind,"#e34a33")

	ind = grep("e2",cn)
	plot(times,as.numeric(as.vector(t[i,ind])),type="n",main=paste("Extinction",A2),
		ylim=c(0,YlimE),ylab="Extinction rate",xlab="Time")
	add_plot(ind,"#e34a33")

	boxplot(mcmc_t[grep("cov_e",colnames(mcmc_t))])
	boxplot(mcmc_t[grep("q2",colnames(mcmc_t))]   , ylim = c(min(mcmc_t[grep("q2",colnames(mcmc_t))]   )*0.7,1.3*max(mcmc_t[grep("q2",colnames(mcmc_t))]   )))
}


A1 = "North America"
A2 = "South America"
f="/Users/danielesilvestro/Downloads/des_juan/North_South_DES_in_rep1_0_q_2.6_5.3_TdD_TdE_ML_marginal_rates.log"
plot_DE_rates(f,A1,A2,n_lines=100,Alpha=0.1,YlimD = 0.2,YlimE = 0.5)


