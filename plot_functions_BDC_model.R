# functions to plot results of BDC model (Silvestro, Warnock et al. 2018 Nature Comm.)

### SKYLINE PLOTS
# PLOT SPECIATION MODE PREVALENCE
plot_polygon <- function(x,t1,t2,color="#3dab3c"){
	library(HDInterval)
	library(scales)
	for (lev in seq(0.95,0.10, length.out =30)){
		hpd = hdi(x,lev)
		polygon(x=-c(t1,t1,t2,t2), y = c(as.numeric(hpd[1:2]),rev(as.numeric(hpd[1:2]))),border=F,col=alpha(color,0.075))	
	}
}


plot_speciation_mode <- function(f,time_bins,Ymax,title=""){
	library(latex2exp)
	tbl = read.table(f,h=T)
	tbl = tbl[50:dim(tbl)[1],]
	
	indx_lambda_foss = grep("lambda_",colnames(tbl))
	indx_lambda_tree = grep("tree_sp",colnames(tbl))
	indx_mu_foss =     grep("mu_",colnames(tbl))
	indx_mu_tree =     grep("tree_ex",colnames(tbl))
	
	dL = tbl[indx_lambda_foss]- 2*tbl[indx_lambda_tree]
	dL2= tbl[indx_lambda_foss]-   tbl[indx_lambda_tree]
	
	time_bins = rev(sort(time_bins))
	
	x = c(1,2,3)
	par(mfrow=c(3,2))
	# speciation mode
	plot(mean(x),xlim=c(-max(time_bins),-min(time_bins)),ylim=c(-Ymax,Ymax),type="n",ylab=TeX('Budding vs anagenetic speciation ($\\lambda^{*} - 2 \\lambda)'),xlab="Time (Ma)")
	abline(h=0,lty=2)
	for (t in 2:length(time_bins)){
		t1=time_bins[t-1]
		t2= time_bins[t]
		x=as.numeric(dL[,t-1])
		plot_polygon(x,t1,t2,color="#3dab3c")
	}

	# anagenetic + bifurcation mode
	plot(mean(x),xlim=c(-max(time_bins),-min(time_bins)),ylim=c(0,Ymax),type="n",
		ylab=TeX('Bifurcation + anagenetic speciation ($\\lambda^{*} - \\lambda)'),xlab="Time (Ma)")
	for (t in 2:length(time_bins)){
		t1=time_bins[t-1]
		t2= time_bins[t]
		x=as.numeric(dL2[,t-1])
		plot_polygon(x,t1,t2,color="#3dab3c")
	}
	
	# speciation rate
	plot(mean(x),xlim=c(-max(time_bins),-min(time_bins)),ylim=c(0,Ymax),type="n",
		ylab=TeX('$\\lambda^{*}'),xlab="Time (Ma)",main="Fossil estimates")
	for (t in 2:length(time_bins)){
		t1=time_bins[t-1]
		t2= time_bins[t]
		x=as.numeric(tbl[indx_lambda_foss][,t-1])
		plot_polygon(x,t1,t2,color="#084594")
	}

	plot(mean(x),xlim=c(-max(time_bins),-min(time_bins)),ylim=c(0,Ymax),type="n",
		ylab=TeX('$\\lambda'),xlab="Time (Ma)",main="Phylogenetic estimates")
	for (t in 2:length(time_bins)){
		t1=time_bins[t-1]
		t2= time_bins[t]
		x=as.numeric(tbl[indx_lambda_tree][,t-1])
		plot_polygon(x,t1,t2,color="#3182bd")
	}
	
	# extinction rate
	plot(mean(x),xlim=c(-max(time_bins),-min(time_bins)),ylim=c(0,Ymax),type="n",ylab=TeX('$\\mu^{*}'),xlab="Time (Ma)")
	for (t in 2:length(time_bins)){
		t1=time_bins[t-1]
		t2= time_bins[t]
		x=as.numeric(tbl[indx_mu_foss][,t-1])
		plot_polygon(x,t1,t2,color="#b30000")
	}
	plot(mean(x),xlim=c(-max(time_bins),-min(time_bins)),ylim=c(0,Ymax),type="n",ylab=TeX('$\\mu'),xlab="Time (Ma)")
	for (t in 2:length(time_bins)){
		t1=time_bins[t-1]
		t2= time_bins[t]
		x=as.numeric(tbl[indx_mu_tree][,t-1])
		plot_polygon(x,t1,t2,color="#e34a33")
	}
	
	
}



pdf("your_path/PyRate_github/example_files/BDC_model/Ferns_BDCskyline_rates.pdf",width=8,height=12 )
time_bins = rev(c(0,25,50,75,100,125,150,175))
f="your_path/PyRate_github/example_files/BDC_model/Ferns_BDCskyline_mcmc.log"
plot_speciation_mode(f,time_bins,0.03)
dev.off()




### CONSTANT RATE MODELS
# PLOT SPECIATION MODE PREVALENCE
plot_speciation_mode <- function(f,title=""){
	library(latex2exp)
	tbl = read.table(f,h=T)
	tbl = tbl[500:dim(tbl)[1],]
	dL = tbl$lambda_0 - 2*tbl$tree_sp
	#hist(dL)
	library(easyGgplot2)
	dL = data.frame(dL)
	colnames(dL)="trait"
	ggplot2.density( dL,xName="trait" ,xlim =c(-0.5,0.5),densityFill="#3dab3c",main=title,ylim=c(0,16))
	
}

plot_bifurcating_plus_anagenetic <- function(f,title=""){
	library(latex2exp)
	tbl = read.table(f,h=T)
	tbl = tbl[500:dim(tbl)[1],]
	dL = tbl$lambda_0 - tbl$tree_sp
	#hist(dL)
	library(easyGgplot2)
	dL = data.frame(dL)
	colnames(dL)="trait"
	ggplot2.density( dL,xName="trait" ,xlim =c(0,0.7),densityFill="#3dab3c",main=title)
	
}

library(latex2exp)
TeX = function(x){return(x)}
plot_fossil_phylo <- function(mcmc_file){
	print(mcmc_file)
	t = read.table(mcmc_file,h=T)
	burnin=round(dim(t)[1]*0.25)

	lambda_0 = t$lambda_0[burnin:dim(t)[1]]
	mu_0     = t$mu_0    [burnin:dim(t)[1]]
	tree_sp  = t$tree_sp [burnin:dim(t)[1]]
	tree_ex  = t$tree_ex [burnin:dim(t)[1]]

	deltaSP = lambda_0-tree_sp
	deltaEX = mu_0-tree_ex

	par(mfrow=c(2,2))

	treeDIV = tree_sp-tree_ex
	fossDIV = lambda_0-mu_0

	#hist(fossDIV-treeDIV)
	comp = (fossDIV-treeDIV)
	length(comp[comp>0])/length(comp)

	library(scales)
	maxR = max(c(lambda_0,mu_0,tree_sp ,tree_ex ))*1.2
	plot(lambda_0~tree_sp,pch=19,col=alpha("blue",0.20), xlim=c(0,maxR),ylim=c(0,maxR),xlab="Phylogenetic estimates",ylab="Fossil estimates",main=strsplit(mcmc_file,"_")[[1]][1])
	points(mu_0~tree_ex,pch=19,col=alpha("red",0.20))
	abline(0,1, lty=2)

	dSP = lambda_0-tree_sp
	dEX = mu_0-tree_ex
	minR = min(c(dSP,dEX,0))*1.2
	maxR = max(c(dSP,dEX))*1.2

	#hist(fossDIV-treeDIV)
	comp = (dSP-dEX)
	P = length(comp[comp>0])/length(comp)

	plot(dSP~dEX,xlim=c(minR,maxR),ylim=c(minR,maxR),pch=19,col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - \\lambda$'),ylab=TeX('$\\mu^{*} - \\mu$'),main=paste("P =",round(P,3)) )
	abline(0,1, lty=2)
	d = sort(lambda_0-2*tree_sp)
	P_neg = length(d[d<0])/length(d)
	Sp_mean = mean(d)
	Sp_min = d[round(length(d)*0.025)]
	Sp_max = d[round(length(d)*0.975)]
	hist((lambda_0-2*tree_sp),col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - 2\\lambda$'), main=paste("P_neg",round(c(P_neg,Sp_mean,Sp_min,Sp_max),2)))

	d2 = mu_0-(tree_sp+tree_ex)

	hist(d2,col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - 2\\lambda$'))


	
	
}
