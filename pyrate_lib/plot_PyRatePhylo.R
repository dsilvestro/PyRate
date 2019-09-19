f = "/Users/danielesilvestro/Dropbox (Personal)/SpeciesConcept/empirical_fossil_clades/pyrate_mcmc_logs/CetaceaBD1-1_mcmc.log"
f = "/Users/danielesilvestro/Dropbox (Personal)/SpeciesConcept/tree_fossil_pyrate/pyrate_mcmc_logs/Ruminant_occurrences_crown_1BD1-1_mcmc.log"
f = "/Users/danielesilvestro/Dropbox (Personal)/SpeciesConcept/empirical_fossil_clades/pyrate_mcmc_logs/Sphenisciformes_1BD1-1_mcmc.log"
f = "/Users/danielesilvestro/Dropbox (Personal)/SpeciesConcept/tree_fossil_pyrate/spconcepts/fern030316_1BD1-1_mcmc.log"

f = "/Users/danielesilvestro/Desktop/tests_software/conifer_Fabien/pyrate_mcmc_logs/Conifers_merged_10_BDMCMC_hppL_G_se_est_0_BD1-1_mcmc.log"	

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
	# minR = min(c(lambda_0-mu_0, 0 ))*2
	# maxR = max(c(lambda_0-mu_0,tree_sp-tree_ex ))*2
	# NDdiff = fossDIV-treeDIV
	# plot(fossDIV~treeDIV,xlim=c(minR,maxR),ylim=c(minR,maxR),pch=19,col=alpha("black",0.20),xlab=TeX('$\\lambda - \\mu$'),ylab=TeX('$\\lambda^{*} - \\mu^{*}$'),main=paste("Net diversification, P =", round(length(NDdiff[NDdiff>0])/length(NDdiff),3) ))

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


plot_species_mode <- function(mcmc_file,maxXY=0.5){
	print(mcmc_file)
	t = read.table(mcmc_file,h=T)
	burnin=round(dim(t)[1]*0.25)

	lambda_0 = t$lambda_0[burnin:dim(t)[1]]
	mu_0     = t$mu_0    [burnin:dim(t)[1]]
	tree_sp  = t$tree_sp [burnin:dim(t)[1]]
	tree_ex  = t$tree_ex [burnin:dim(t)[1]]

	deltaSP = lambda_0-tree_sp
	deltaEX = mu_0-tree_ex

	d1 = lambda_0-2*tree_sp
	d2 = mu_0-(tree_sp+tree_ex)
	max_d = maxXY #max(abs(c(d1,d2)))                      
	plot(d1~d2,xlim=c(-max_d,max_d),ylim=c(-max_d,max_d),pch=19,col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - 2\\lambda$'),
	ylab=TeX('$\\mu^{*} - (\\lambda + \\mu$)'),main=strsplit(mcmc_file,"_")[[1]][1])
	abline(v=0, lty=2)
	abline(h=0, lty=2)
		
}

plot_fossil_phylo(f)

setwd("/Users/danielesilvestro/Dropbox_Personal/SpeciesConcept/tree_fossil_pyrate/spconcepts")
files = list.files(path = "/Users/danielesilvestro/Dropbox_Personal/SpeciesConcept/tree_fossil_pyrate/spconcepts", pattern ="mcmc.log")

setwd("/Users/danielesilvestro/Documents/Projects/SpeciesConcept/tree_fossil_pyrate/spconcepts")
files = list.files(path = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/tree_fossil_pyrate/spconcepts", pattern ="mcmc.log")


setwd("/Volumes/dsilvestro/Species_Concept/sampling_simulations/sim3")
files = list.files(path = "/Volumes/dsilvestro/Species_Concept/sampling_simulations/sim3", pattern ="mcmc.log")



pdf(file="output_BDC.pdf",width=10,height=10)

par(mfrow=c(2,2))
for (i in 1:length(files)){
	plot_fossil_phylo(subset_files[i])
}
dev.off()




subset_files = files[c(1,3,4,5)]
pdf(file="output_BDC2m.pdf",width=10,height=10)

#par(mfrow=c(2,2))
for (i in 1:length(subset_files)){
	#plot_species_mode(subset_files[i],0.5)
	plot_fossil_phylo(subset_files[i])
}
dev.off()




# TEST whether model 1 is significantly rejected
setwd("/Users/danielesilvestro/Dropbox_Personal/SpeciesConcept/tree_fossil_pyrate/spconcepts")
files = list.files(path = "/Users/danielesilvestro/Dropbox_Personal/SpeciesConcept/tree_fossil_pyrate/spconcepts", pattern ="mcmc.log")
compare_fossil_phylo <- function(mcmc_file){
	print(mcmc_file)
	#mcmc_file = "/Users/danielesilvestro/Dropbox (Personal)/SpeciesConcept/tree_fossil_pyrate/spconcepts/Canidae_1BD1-1_mcmc.log"
	t = read.table(mcmc_file,h=T)
	burnin=round(dim(t)[1]*0.25)

	lambda_0 = t$lambda_0[burnin:dim(t)[1]]
	mu_0     = t$mu_0    [burnin:dim(t)[1]]
	tree_sp  = t$tree_sp [burnin:dim(t)[1]]
	tree_ex  = t$tree_ex [burnin:dim(t)[1]]
		
	rel_errSP = mean((tree_sp - lambda_0)/lambda_0)
	rel_errEX = mean((tree_ex - mu_0)/mu_0)
	
	max_fold_differenceSP = max(c(mean(tree_sp)/mean(lambda_0), 1/(mean(tree_sp)/mean(lambda_0))))
	max_fold_differenceEX = max(c(mean(tree_ex)/mean(mu_0), 1/(mean(tree_ex)/mean(mu_0))))
	
	deltaSP = lambda_0-tree_sp
	deltaEX = mu_0-tree_ex
	
	comp = (deltaSP-deltaEX)
	P_model1sp = length(deltaSP[deltaSP>0])/length(deltaSP)
	P_model1ex = length(deltaEX[deltaEX>0])/length(deltaEX)
	P_model1spex = length(intersect(which(deltaSP<0),which(deltaEX<0)))/length(deltaEX)
	
	eqn5 = (lambda_0-tree_sp) - (mu_0-tree_ex)
	P_model2 = length(eqn5[eqn5>0])/length(eqn5)
	
	print( round(c(P_model1sp,P_model1ex,P_model1spex,P_model2,mean(abs(c(rel_errSP,rel_errEX))),max_fold_differenceSP,max_fold_differenceEX),3))
	return(c(P_model1sp,P_model1ex,P_model1spex,P_model2,mean(abs(c(rel_errSP,rel_errEX))),max_fold_differenceSP,max_fold_differenceEX))
	
}

res = NULL
for (i in 1:length(files)){
	res = rbind(res,compare_fossil_phylo(files[i]))
}

#__ [1] "Bovidae_1BD1-1_mcmc.log"
#__ [1]  1.000  1.000  0.000  0.039  0.829  3.251 30.168
#__ [1] "Canidae_1BD1-1_mcmc.log"
#__ [1] 0.954 0.952 0.023 0.372 0.472 1.452 2.792
#__ [1] "Cervidae_1BD1-1_mcmc.log"
#__ [1] 0.993 0.998 0.002 0.019 0.600 1.732 4.570
#__ [1] "Cetacea_0523_1BD1-1_mcmc.log"
#__ [1]  1.000  1.000  0.000  0.017  0.772  2.819 10.005
#__ [1] "Feliformia_1BD1-1_mcmc.log"
#__ [1] 1.000 1.000 0.000 0.771 0.690 2.332 5.290
#__ [1] "fern030316_1BD1-1_mcmc.log"
#__ [1]  1.000  1.000  0.000  0.000  0.628  1.518 11.954
#__ [1] "Ferns_1BD1-1_mcmc.log"
#__ [1] 0.534 1.000 0.000 0.000 0.437 1.008 7.902
#__ [1] "Scleractinia_sp_1BD1-1_mcmc.log"
#__ [1] 0.000 0.000 1.000 0.405 3.426 4.309 4.541
#__ [1] "Sphenisciformes_1BD1-1_mcmc.log"
#__ [1] 0.202 0.475 0.514 0.105 0.347 1.500 1.139
#__ [1] "Ursidae_1BD1-1_mcmc.log"
#__ [1] 0.922 0.940 0.052 0.314 0.529 1.873 2.520


# PLOT PHYLO -FOSSIL under SKYLINE (independent) MODEL
library(latex2exp)

plot_fossil_phylo_skyline <- function(mcmc_file,burnin=0.2,maxR_arg=0){
	print(mcmc_file)
	t = read.table(mcmc_file,h=T)
	burnin=round(dim(t)[1]*0.5)
	
	col_lambda  = grep("lambda_",colnames(t))
	col_mu      = grep("mu_",colnames(t))
	col_tree_sp = grep("tree_sp",colnames(t))
	col_tree_ex = grep("tree_ex",colnames(t))
	
	for (ind in 1:length(col_lambda)){
		lambda_0 = t[burnin:dim(t)[1], col_lambda[ind] ]
		mu_0     = t[burnin:dim(t)[1], col_mu[ind] ]
		tree_sp  = t[burnin:dim(t)[1], col_tree_sp[ind] ]
		tree_ex  = t[burnin:dim(t)[1], col_tree_ex[ind] ]

		deltaSP = lambda_0-tree_sp
		deltaEX = mu_0-tree_ex

		#par(mfrow=c(2,2))

		treeDIV = tree_sp-tree_ex
		fossDIV = lambda_0-mu_0

		#hist(fossDIV-treeDIV)
		comp = (fossDIV-treeDIV)
		length(comp[comp>0])/length(comp)

		library(scales)
		if (maxR_arg==0){
			#maxR = max(c(lambda_0,mu_0,tree_sp ,tree_ex ))*1.2
			maxR = sort(c(lambda_0,mu_0,tree_sp ,tree_ex ))[round(0.99*length(tree_ex)*4)]     *1.2
		}else{
			maxR = maxR_arg
		}

		names_bin = c(">150 Ma","150-125 Ma","125-100 Ma","100-75 Ma","75-50 Ma","50-25 Ma","25-0 Ma")
		
		plot(lambda_0~tree_sp,pch=19,col=alpha("blue",0.20), xlim=c(0,maxR),ylim=c(0,maxR),xlab="Phylogenetic estimates",
			ylab="Fossil estimates",main= names_bin[ind]) #colnames(t)[col_lambda[ind]])
		points(mu_0~tree_ex,pch=19,col=alpha("red",0.20))
		abline(0,1, lty=2)
		# minR = min(c(lambda_0-mu_0, 0 ))*2
		# maxR = max(c(lambda_0-mu_0,tree_sp-tree_ex ))*2
		# NDdiff = fossDIV-treeDIV
		# plot(fossDIV~treeDIV,xlim=c(minR,maxR),ylim=c(minR,maxR),pch=19,col=alpha("black",0.20),xlab=TeX('$\\lambda - \\mu$'),ylab=TeX('$\\lambda^{*} - \\mu^{*}$'),main=paste("Net diversification, P =", round(length(NDdiff[NDdiff>0])/length(NDdiff),3) ))

		dSP = lambda_0-tree_sp
		dEX = mu_0-tree_ex
		minR = min(c(dSP,dEX,0))*1.2
		maxR = max(c(dSP,dEX))*1.2

		#hist(fossDIV-treeDIV)
		comp = (dSP-dEX)
		P = length(comp[comp>0])/length(comp)

		plot(dSP~dEX,xlim=c(minR,maxR),ylim=c(minR,maxR),pch=19,col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - \\lambda$'),ylab=TeX('$\\mu^{*} - \\mu$'),main=paste("P =",round(min(P,1-P),3)) )
		abline(0,1, lty=2)
		
		hist((lambda_0-tree_sp),col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - \\lambda$'),
				 main=paste("P(l* > l) =",round(length(which(tree_sp<lambda_0))/length(lambda_0),2)))
		
 		hist((mu_0-tree_ex),col=alpha("black",0.20),xlab=TeX('$\\mu^{*} - \\mu$'),
 				 main=paste("P(m* > m) =",round(length(which(tree_ex<mu_0))/length(mu_0),2)))
		
		#d = sort(lambda_0-2*tree_sp)
		#P_neg = length(d[d<0])/length(d)
		#Sp_mean = mean(d)
		#Sp_min = d[round(length(d)*0.025)]
		#Sp_max = d[round(length(d)*0.975)]
		#hist((lambda_0-2*tree_sp),col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - 2\\lambda$'),
		# main=paste("P_neg",round(c(P_neg,Sp_mean,Sp_min,Sp_max),2)))
            #
		#d2 = mu_0-(tree_sp+tree_ex)
            #
		#hist(d2,col=alpha("black",0.20),xlab=TeX('$\\mu^{*} - (\\lambda + \\mu  )$'))
		
	}
}

pdf(file="/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern_BDCskyline.pdf",width=10,height=10)
mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern030316_125myr_priorBD1-1_mcmc.log"
mcmc_file = "/Users/danielesilvestro/Downloads/paleobioDB_turtles_1_GBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file,0.1)
dev.off()
mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern030316_125myrBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file)
mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern030316_125myr_newpropBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file)

mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern030316_110myr_priorBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file)
mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern030316_110myr_newpropBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file)

pdf(file="/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/coral_BDCskyline.pdf",width=10,height=10)
mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/Scleractinia_sp_125myrBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file,0.9)
dev.off()


# PLOT PHYLO -FOSSIL under SKYLINE (independent) MODEL ONE PAGE
## library(latex2exp)

plot_fossil_phylo_skyline <- function(mcmc_file,burnin=0.2,maxR_arg=0){
	print(mcmc_file)
	t = read.table(mcmc_file,h=T)
	burnin=round(dim(t)[1]*0.5)
	
	col_lambda  = grep("lambda_",colnames(t))
	col_mu      = grep("mu_",colnames(t))
	col_tree_sp = grep("tree_sp",colnames(t))
	col_tree_ex = grep("tree_ex",colnames(t))
	
	for (ind in 1:length(col_lambda)){
		lambda_0 = t[burnin:dim(t)[1], col_lambda[ind] ]
		mu_0     = t[burnin:dim(t)[1], col_mu[ind] ]
		tree_sp  = t[burnin:dim(t)[1], col_tree_sp[ind] ]
		tree_ex  = t[burnin:dim(t)[1], col_tree_ex[ind] ]

		deltaSP = lambda_0-tree_sp
		deltaEX = mu_0-tree_ex

		treeDIV = tree_sp-tree_ex
		fossDIV = lambda_0-mu_0

		#hist(fossDIV-treeDIV)
		comp = (fossDIV-treeDIV)
		length(comp[comp>0])/length(comp)

		library(scales)
		if (maxR_arg==0){
			#maxR = max(c(lambda_0,mu_0,tree_sp ,tree_ex ))*1.2
			maxR = sort(c(lambda_0,mu_0,tree_sp ,tree_ex ))[round(0.99*length(tree_ex)*4)]     *1.2
		}else{
			maxR = maxR_arg
		}

		names_bin = c(">150 Ma","150-125 Ma","125-100 Ma","100-75 Ma","75-50 Ma","50-25 Ma","25-0 Ma")
		
		plot(lambda_0~tree_sp,pch=19,col=alpha("blue",0.20), xlim=c(0,maxR),ylim=c(0,maxR),xlab="Phylogenetic estimates",
			ylab="Fossil estimates",main= names_bin[ind]) #colnames(t)[col_lambda[ind]])
		points(mu_0~tree_ex,pch=19,col=alpha("red",0.20))
		abline(0,1, lty=2)

		dSP = lambda_0-tree_sp
		dEX = mu_0-tree_ex
		minR = min(c(dSP,dEX,0))*1.2
		maxR = max(c(dSP,dEX))*1.2

		#hist(fossDIV-treeDIV)
		comp = (dSP-dEX)
		P = length(comp[comp>0])/length(comp)

		plot(dSP~dEX,xlim=c(minR,maxR),ylim=c(minR,maxR),pch=19,col=alpha("black",0.20),xlab=TeX('$\\lambda^{*} - \\lambda$'),ylab=TeX('$\\mu^{*} - \\mu$'),main=paste("P =",round(min(P,1-P),3)) )
		abline(0,1, lty=2)
		
	}
}

pdf(file="/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern_BDCskylineONEPAGE.pdf",width=12,height=12)
par(mfrow=c(4,4))
mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/fern030316_125myr_priorBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file,0.1)
dev.off()

pdf(file="/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/coral_BDCskylineONEPAGE.pdf",width=12,height=12)
par(mfrow=c(4,4))
mcmc_file = "/Users/danielesilvestro/Documents/Projects/SpeciesConcept/final_empirical_data/fern_skyline/OTHER_pyrate_mcmc_logs/Scleractinia_sp_125myrBD1-1_mcmc.log"
plot_fossil_phylo_skyline(mcmc_file,0.1)
dev.off()
