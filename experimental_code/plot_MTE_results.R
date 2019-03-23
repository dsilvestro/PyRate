library(scales)

get_thresholds <- function(prior_pr){
	prior_odds = (0.05/(1-0.05)) 
	BF = c(10,6,2)
	A = exp(BF/2) * prior_odds
	thr = A/ (A+1)
	return(thr)
}

plot_mte <- function(logfile,color_trait=c(),prior_prob = 0.05){
	
	pdf(file=paste(sub(".log","",logfile,fixed = T),".pdf",sep=""),width=10, height=7)
	par(mfrow=c(2,3))
	
	tbl = read.table(logfile,h=T)
	indx = grep("I_",colnames(tbl))
	trait_names = gsub("I_", "", colnames(tbl)[indx])

	if (length(color_trait)==0){
		color_trait = rep("#08306b",length(trait_names))
	}
	
	print_name= trait_names
	significance = get_thresholds(prior_prob)
	
	for (trait_i in 1:length(trait_names)){
		indx_trait = grep(paste("m_",trait_names[trait_i],sep=""),colnames(tbl))
		
		indicator = tbl[,indx[trait_i]]
		title = sprintf("%s (P = %s)", print_name[trait_i], round(mean(indicator),2))
		
		if (mean(indicator) > significance[1]){
			alphaC = 1
		}else if (mean(indicator) > significance[2]){
			alphaC = 0.85
		}else if (mean(indicator) > significance[3]){
			alphaC = 0.4
		}else{
			alphaC = 0.1
		}
		
		tbl_red = tbl[,indx_trait]
		colnames(tbl_red)= gsub(paste("m_",trait_names[trait_i],"_",sep=""),"", colnames(tbl_red)  )
		# col=c("#ffffb2","#fd8d3c","#f03b20","#bd0026")
		boxplot(tbl_red,ylab="Relative effect on extinction",outline=F,main=title,xlab=trait_names[trait_i],notch=T,plot=T,col=alpha(color_trait[trait_i],alphaC),lty=1)
		abline(h=1/length(indx_trait),lty=2)
	
	}
	n <- dev.off()
}


color_trait = c("#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f")

#f = "mte_output.log"
#plot_mte(f,color_trait)
