library(stats4)
library(scales)
library(HDInterval)

### PLOT EMPIRICAL
add_geochrono <- function(Y1,Y2){	
	polygon(-c(259.1,259.1,251.9,251.9), c(Y1,Y2,Y2,Y1), col = "#f9b4a3", lwd = 0.5) # Lower Permian
	polygon(-c(251.9,251.9,247.2,247.2), c(Y1,Y2,Y2,Y1), col = "#a05da5", lwd = 0.5) # Upper Triassic	
	polygon(-c(247.2,247.2,237,237),     c(Y1,Y2,Y2,Y1), col = "#b282ba", lwd = 0.5) # Middle Triassic
	polygon(-c(237,237,201.3,201.3),     c(Y1,Y2,Y2,Y1), col = "#bc9dca", lwd = 0.5) # Upper Triassic
	polygon(-c(201.3,201.3,174.1,174.1), c(Y1,Y2,Y2,Y1), col = "#00b4eb", lwd = 0.5) # Lower Jurassic
	polygon(-c(174.1,174.1,163.5,163.5), c(Y1,Y2,Y2,Y1), col = "#71cfeb", lwd = 0.5) # Middle Jurassic
	polygon(-c(163.5,163.5,145,145),     c(Y1,Y2,Y2,Y1), col = "#abe1fa", lwd = 0.5) # Upper Jurassic
	polygon(-c(145,145,100.5,100.5),     c(Y1,Y2,Y2,Y1), col = "#A0C96D", lwd = 0.5) # Lower Cretaceous
	polygon(-c(100.5,100.5,66,66),       c(Y1,Y2,Y2,Y1), col = "#BAD25F", lwd = 0.5) # Upper Cretaceous
	polygon(-c(66,66,56,56),             c(Y1,Y2,Y2,Y1), col = "#F8B77D", lwd = 0.5) # Paleocene
	polygon(-c(56,56,33.9,33.9),         c(Y1,Y2,Y2,Y1), col = "#FAC18A", lwd = 0.5) # Eocene
	polygon(-c(33.9,33.9,23.03,23.03),   c(Y1,Y2,Y2,Y1), col = "#FBCC98", lwd = 0.5) # Oligocene
	polygon(-c(23.03,23.03,5.33,5.33),   c(Y1,Y2,Y2,Y1), col = "#FFED00", lwd = 0.5) # Miocene
	polygon(-c(5.33,5.33,2.58,2.58),     c(Y1,Y2,Y2,Y1), col = "#FFF7B2", lwd = 0.5) # Pliocene
	polygon(-c(2.58,2.58,0.0117,0.0117), c(Y1,Y2,Y2,Y1), col = "#FFF1C4", lwd = 0.5) # Pleistocene
	polygon(-c(0.0117,0.0117,0,0),       c(Y1,Y2,Y2,Y1), col = "#FEF6F2", lwd = 0.5) # Holocene
}

plot_polygon <- function(x,t1,t2,color){
	for (lev in seq(0.95,0.10, length.out =10)){
		hpd = hdi(x,lev)
		polygon(x=-c(t1,t1,t2,t2), y = c(as.numeric(hpd[1:2]),rev(as.numeric(hpd[1:2]))),border=F,col=alpha(color,0.075))	
	}
}

plot_diversity <- function(log_file,col_alpha=0.2,skyline_plot=0, color="#3dab3c", title=""){
	tbl_post = read.table(log_file, h=T)
	tbl_post_burnin = tbl_post[10:dim(tbl_post)[1],]
	tbl = t(as.matrix(tbl_post_burnin[7:dim(tbl_post_burnin)[2]]))

	# PARSE AGES
	ages = as.numeric(gsub("t_","",row.names(tbl)))
		
	# get relative diversity
	tbl_frac = t(t(tbl[,-1]))
	P_predicted_diversity_mean = apply(tbl_frac, FUN=mean,1)

	ages = -ages

	Ylab= "Diversity"		

	hpd50 = hdi(t(tbl_frac),0.50)
	hpd75 = hdi(t(tbl_frac),0.75)
	hpd95 = hdi(t(tbl_frac),0.99)
	minY = min(hpd95)
	maxY = max(hpd95)
	
	plot(ages,P_predicted_diversity_mean,type="n",main=title,xlab="Time (Ma)",ylab = Ylab, ylim=c(minY,maxY),xlim = c(min(ages),max(ages)))	
	if (skyline_plot==1){
		x1 = -ages
		age_m = x1+c(diff(x1)/2,mean(diff(x1)/2))
		ages_m = c(age_m[1:length(age_m)],max(x1))
		for (i in 2:length(ages)){
			plot_polygon(t(tbl_frac)[,i],ages_m[i],ages_m[i-1],color )
			add_geochrono(0, -0.05*(maxY-minY))
		}
	}else{
		hpd_list = list(hpd95,hpd75,hpd50)
		colors = c("#cccccc","#969696","#525252")
		colors = c("#7fc97f","#7fc97f","#7fc97f")
		for (i in 1:length(hpd_list)){
			hpd_temp = hpd_list[[i]]
			polygon(c(ages, rev(ages)), c(hpd_temp[1,], rev(hpd_temp[2,])), col = alpha(colors[i],col_alpha), border = NA)
		}
		lines(ages,P_predicted_diversity_mean,col="#7fc97f",lwd=3)		
		add_geochrono(0, -0.05*(maxY-minY))
	}
	
}





