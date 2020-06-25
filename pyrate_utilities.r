no.extension <- function(filename) { 
	  if (substr(filename, nchar(filename), nchar(filename))==".") { 
	    return(substr(filename, 1, nchar(filename)-1)) 
	  } else { 
	    no.extension(substr(filename, 1, nchar(filename)-1)) 
	  } 
	}


extract.ages <- function(file = NULL, replicates = 1, cutoff = NULL, random = TRUE, outname = "_PyRate",save_tax_list=T){

	if (is.null(file)) 
	    stop("you must enter a filename or a character string\n")

	rnd <- random
	q <- cutoff
	dat1 <- read.table(file, header=T, stringsAsFactors=F, row.names=NULL, sep="\t", strip.white=T)
	fname <- no.extension(basename(file))
	outfile <- paste(dirname(file), "/", fname, outname, ".py", sep="")

	dat1[,1] <- gsub("[[:blank:]]{1,}","_", dat1[,1])

	if (replicates > 1){
		rnd <- TRUE
	}

	if (any(is.na(dat1[,1:4]))){
		print(c(which(is.na(dat1[,1])),which(is.na(dat1[,2])),which(is.na(dat1[,3])),which(is.na(dat1[,4]))  )   )
		stop("the input file contains missing data in species names, status or ages)\n")
	}

	if (!is.null(q)){
			dat <- dat1[!(dat1[,4] - dat1[,3] >= q),]
			cat("\n\nExcluded ", 100-round(100*dim(dat)[1]/dim(dat1)[1]), "% occurrences")
			hist(dat1[,4] - dat1[,3])
		} else { 
			dat <- dat1 
	}

	if (length(dat) == 5 & !("SITE" %in% toupper(colnames(dat)))){
		colnames(dat) <- c("Species", "Status", "min_age", "max_age", "trait")	
	} 
	if (length(dat) == 5 & ("SITE" %in% toupper(colnames(dat)))){
		colnames(dat) <- c("Species", "Status", "min_age", "max_age", "site")	
	} 
	if (length(dat) == 4) {
		colnames(dat) <- c("Species", "Status", "min_age", "max_age")
	}
	dat$new_age <- "NA"
	splist <- unique(dat[,c(1,2)])[order(unique(dat[,c(1,2)][,1])),]


	if (any(is.element(splist$Species[tolower(splist$Status) == "extant"], splist$Species[tolower(splist$Status) == "extinct"]))){
		print(intersect(splist$Species[tolower(splist$Status) == "extant"], splist$Species[tolower(splist$Status) == "extinct"]))
		stop("at least one species is listed as both extinct and extant\n")
	}

	cat("#!/usr/bin/env python", "from numpy import * ", "",  file=outfile, sep="\n")
	
        # if("site" %in% colnames(dat)){}
	for (j in 1:replicates){
		times <- list()
		cat ("\nreplicate", j)
	
                # dat[dat$min_age == 0,3] <- 0.001
	
		#if (any(dat[,4] < dat[,3])){
		#	cat("\nWarning: the min age is older than the max age for at least one record\n")
		#	cat ("\nlines:",1+as.numeric(which(dat[,4] < dat[,3])),sep=" ")
		#}
		if("site" %in% colnames(dat)){
			print("By-site age randomization...")
			# Juan Cantalapiedra's function
			maxima <- aggregate(as.numeric(dat[,3]), by=list(dat$site),mean)[,2]
			minima <- aggregate(as.numeric(dat[,4]), by=list(dat$site),mean)[,2]
                        
                        minima_1 <- apply(cbind(maxima, minima),FUN=min,1)
                        maxima_1 <- apply(cbind(maxima, minima),FUN=max,1)
                        
			rnd_ages <- apply(cbind(minima_1, maxima_1),1, function(x){round(runif(1,x[1],x[2]), digits = 6)})
			names(rnd_ages) <- aggregate(dat[,3], by=list(dat$site),mean)[,1]
			dat$new_age <- rnd_ages[match(dat$site,names(rnd_ages))]
		}else{
			if (isTRUE(rnd)){
				dat$new_age <- round(runif(length(dat[,1]), min=apply(dat[,3:4],FUN=min,1), max=apply(dat[,3:4],FUN=max,1)), digits=6)
			} else {
				# Mark's suggestion
				dat$new_age <- apply(dat[,3:4], 1, mean)	
			}
		}
		
	

		dat2 <- subset(dat, select=c("Species","new_age"))
		taxa <- sort(unique(dat2$Species))

		for (n in 1:length(taxa)){
			times[[n]] <- dat2$new_age[dat2$Species == taxa[n]]
			if (toupper(splist$Status[splist$Species == taxa[n]]) == toupper("extant")){
				times[[n]] <- append(times[[n]], "0", after=length(times[[n]]))
			}
		}

		dat3 <- matrix(data=NA, nrow=length(times), ncol=max(sapply(times, length)))
		rownames(dat3) <- taxa

		for (p in 1:length(times)){
			dat3[p,1:length(times[[p]])] <- times[[p]]
		}

		cat(noquote(sprintf("\ndata_%s=[", j)), file=outfile, append=TRUE)

		for (n in 1:(length(taxa)-1)){
			rec <- paste(dat3[n,!is.na(dat3[n,])], collapse=",")
			cat(noquote(sprintf("array([%s]),", rec)), file=outfile, append=TRUE, sep="\n")
		}

		n <- n+1
		rec <- paste(dat3[n,!is.na(dat3[n,])], collapse=",")
		cat(noquote(sprintf("array([%s])", rec)), file=outfile, append=TRUE, sep="\n")

		cat("]", "", file=outfile, append=TRUE, sep="\n")
	}


	data_sets <- ""
	names <- ""

	if (replicates > 1){
		for (j in 1:(replicates-1)) {
			data_sets <- paste(data_sets, noquote(sprintf("data_%s,", j)))
			names <- paste(names, noquote(sprintf(" '%s_%s',", fname,j)))
			}

		data_sets <- paste(data_sets, noquote(sprintf("data_%s", j+1)))
		names <- paste(names, noquote(sprintf(" '%s_%s',", fname,j+1)))
	} else {
		data_sets <- "data_1"
		names <- noquote(sprintf(" '%s_1'", fname))	
	}

	cat(noquote(sprintf("d=[%s]", data_sets)), noquote(sprintf("names=[%s]", names)), "def get_data(i): return d[i]", "def get_out_name(i): return  names[i]", file=outfile, append=TRUE, sep="\n")


	tax_names <- paste(taxa, collapse="','")
	cat(noquote(sprintf("taxa_names=['%s']", tax_names)), "def get_taxa_names(): return taxa_names", file=outfile, append=TRUE, sep="\n")


	if ("trait" %in% colnames(dat)){
		datBM <- dat[,1]
		splist$Trait <- NA
		for (n in 1:length(splist[,1])){
			splist$Trait[n] <- mean(dat$trait[datBM == splist[n,1]], na.rm=T)
		}
		s1 <- "\ntrait1=array(["
		BM <- gsub("NaN|NA", "nan", toString(splist$Trait))
		s2 <- "])\ntraits=[trait1]\ndef get_continuous(i): return traits[i]"
		STR <- paste(s1,BM,s2)
		cat(STR, file=outfile, append=TRUE, sep="\n")
	}

	if (save_tax_list == T){
		splistout <- paste(dirname(file), "/", fname, "_TaxonList.txt", sep="")
		lookup <- as.data.frame(taxa)
		lookup$status  <- "extinct"
		write.table(splist, file=splistout, sep="\t", row.names=F, quote=F)	
	}
	cat("\n\nPyRate input file was saved in: ", sprintf("%s", outfile), "\n\n")

}


fit.prior <- function(file = NULL, lineage = "root_age"){

	require(fitdistrplus)

	if (is.null(file)){
	    stop("You must enter a valid filename.\n")
		}

	dat <- read.table(file, header=T, stringsAsFactors=F, row.names=NULL, sep="\t")
	fname <- no.extension(basename(file))
	outfile <- paste(dirname(file), "/", lineage, "_Prior.txt", sep="")

	lineage2 <- paste(lineage,"_TS", sep="")
	if (!is.element(lineage2, colnames(dat))){
		stop("Lineage not found, please check your input.\n")
		}

	time <- dat[,which(names(dat) == lineage2)]
	time2 <- time-(min(time)-0.01)
	gamm <- fitdist(time2, distr="gamma", method = "mle")$estimate 

	cat("Lineage: ", lineage, "; Shape: ", gamm[1], "; Scale: ", 1/gamm[2], "; Offset: ", min(time), sep="", file=outfile, append=FALSE)
}




extract.ages.pbdb <- function(file = NULL,sep=",", extant_species = c(), replicates = 1, cutoff = NULL, random = TRUE){
	print("This function is currently being tested - caution with the results!")
	tbl = read.table(file=file,h=T,sep=sep,stringsAsFactors =F)
	new_tbl = NULL # ADD EXTANT SPECIES
	
	for (i in 1:dim(tbl)[1]){
		if (tbl$accepted_name[i] %in% extant_species){
			status="extant"
		}else{status="extinct"}
		species_name = gsub(" ", "_", tbl$accepted_name[i])
		new_tbl = rbind(new_tbl,c(species_name,status,tbl$min_ma[i],tbl$max_ma[i]))
	}
	colnames(new_tbl) = c("Species","Status","min_age","max_age")
	
	output_file = file.path(dirname(file),strsplit(basename(file), "\\.")[[1]][1])
	output_file = paste(output_file,".txt",sep="")
	write.table(file=output_file,new_tbl,quote=F,row.names = F,sep="\t")
	extract.ages(file=output_file,replicates = replicates, cutoff = cutoff, random = random)
}


extract.ages.tbl <- function(file = NULL,sep="\t", extant_species = c(), replicates = 1, cutoff = NULL, random = TRUE){
	tbl = read.table(file=file,h=T,sep=sep,stringsAsFactors =F)
		
	new_tbl = NULL # ADD EXTANT SPECIES
	
	for (i in 1:dim(tbl)[1]){
		if (tbl[i,1] %in% extant_species){
			status="extant"
		}else{status="extinct"}
		species_name = gsub(" ", "_", tbl[i,1])
		new_tbl = rbind(new_tbl,c(species_name,status,tbl[i,2:dim(tbl)[2]]))
	}
	if (dim(new_tbl)[2]==4){
		colnames(new_tbl) = c("Species","Status","min_age","max_age")
	}else{
		colnames(new_tbl) = c("Species","Status","min_age","max_age","trait")
	}
	
	
	output_file = file.path(dirname(file),strsplit(basename(file), "\\.")[[1]][1])
	output_file = paste(output_file,".txt",sep="")
	write.table(file=output_file,new_tbl,quote=F,row.names = F,sep="\t")
	extract.ages(file=output_file,replicates = replicates, cutoff = cutoff, random = random)
}

extract.ages.14C <- function(file,outname = "_PyRate"){
	dat <- read.table(file, header=T, stringsAsFactors=F, row.names=NULL, sep="\t", strip.white=T)
	fname <- no.extension(basename(file))
	outfile <- paste(dirname(file), "/", fname, outname, ".py", sep="")

	dat[,1] <- gsub("[[:blank:]]{1,}","_", dat[,1])


	colnames(dat)[1] = "Lineage"
	colnames(dat)[2] = "Status"
	dat$new_age <- "NA"
	splist <- unique(dat[,c(1,2)])[order(unique(dat[,c(1,2)][,1])),]


	if (any(is.element(splist$Species[tolower(splist$Status) == "extant"], splist$Species[tolower(splist$Status) == "extinct"]))){
		print(intersect(splist$Species[tolower(splist$Status) == "extant"], splist$Species[tolower(splist$Status) == "extinct"]))
		stop("at least one species is listed as both extinct and extant\n")
	}

	cat("#!/usr/bin/env python", "from numpy import * ", "",  file=outfile, sep="\n")
	
	replicates = dim(dat)[2]-2
	for (j in 1:replicates){
		times <- list()
		cat ("\nreplicate", j)
	
		dat$new_age = dat[,j+2]

		dat2 <- subset(dat, select=c("Lineage","new_age"))
		taxa <- sort(unique(dat2$Lineage))

		for (n in 1:length(taxa)){
			times[[n]] <- dat2$new_age[dat2$Lineage == taxa[n]]
			if (toupper(splist$Status[splist$Lineage == taxa[n]]) == toupper("extant")){
				times[[n]] <- append(times[[n]], "0", after=length(times[[n]]))
			}
		}

		dat3 <- matrix(data=NA, nrow=length(times), ncol=max(sapply(times, length)))
		rownames(dat3) <- taxa

		for (p in 1:length(times)){
			dat3[p,1:length(times[[p]])] <- times[[p]]
		}

		cat(noquote(sprintf("\ndata_%s=[", j)), file=outfile, append=TRUE)

		for (n in 1:(length(taxa)-1)){
			rec <- paste(dat3[n,!is.na(dat3[n,])], collapse=",")
			cat(noquote(sprintf("array([%s.]),", rec)), file=outfile, append=TRUE, sep="\n")
		}

		n <- n+1
		rec <- paste(dat3[n,!is.na(dat3[n,])], collapse=",")
		cat(noquote(sprintf("array([%s.])", rec)), file=outfile, append=TRUE, sep="\n")

		cat("]", "", file=outfile, append=TRUE, sep="\n")
	}


	data_sets <- ""
	names <- ""

	if (replicates > 1){
		for (j in 1:(replicates-1)) {
			data_sets <- paste(data_sets, noquote(sprintf("data_%s,", j)))
			names <- paste(names, noquote(sprintf(" '%s_%s',", fname,j)))
			}

		data_sets <- paste(data_sets, noquote(sprintf("data_%s", j+1)))
		names <- paste(names, noquote(sprintf(" '%s_%s',", fname,j+1)))
	} else {
		data_sets <- "data_1"
		names <- noquote(sprintf(" '%s_1'", fname))	
	}

	cat(noquote(sprintf("d=[%s]", data_sets)), noquote(sprintf("names=[%s]", names)), "def get_data(i): return d[i]", "def get_out_name(i): return  names[i]", file=outfile, append=TRUE, sep="\n")


	tax_names <- paste(taxa, collapse="','")
	cat(noquote(sprintf("taxa_names=['%s']", tax_names)), "def get_taxa_names(): return taxa_names", file=outfile, append=TRUE, sep="\n")

  	cat("\n\nPyRate input file was saved in: ", sprintf("%s", outfile), "\n\n")

}
