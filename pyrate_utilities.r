no.extension <- function(filename) { 
  if (substr(filename, nchar(filename), nchar(filename))==".") { 
    return(substr(filename, 1, nchar(filename)-1)) 
  } else { 
    no.extension(substr(filename, 1, nchar(filename)-1)) 
  } 
}


extract.ages <- function(file = NULL, replicates = 1, cutoff = NULL, random = TRUE){

if (is.null(file)) 
    stop("you must enter a filename or a character string\n")

rnd <- random
q <- cutoff
dat1 <- read.table(file, header=T, stringsAsFactors=F, row.names=NULL, sep="\t")
fname <- no.extension(basename(file))
outfile <- paste(dirname(file), "/", fname, "_PyRate.py", sep="")

if (replicates > 1){
	rnd <- TRUE
}

if (any(is.na(dat1[,1:4]))){
	stop("missing data. Please check your input file.\n")
}

if (!is.null(q)){
		dat <- dat1[!(dat1[,4] - dat1[,3] >= q),]
	} else { 
		dat <- dat1 
}

if (length(dat) == 5){
	colnames(dat) <- c("Species", "Status", "min_age", "max_age", "trait")	
	} else {
	colnames(dat) <- c("Species", "Status", "min_age", "max_age")
}

dat$new_age <- "NA"
splist <- unique(dat[,c(1,2)])[order(unique(dat[,c(1,2)][,1])),]


if (any(is.element(splist$Species[splist$Status == "extant"], splist$Species[splist$Status == "extinct"]))){
	stop("at least one species is listed as both extinct and extant\n")
}

cat("#!/usr/bin/env python", "from numpy import * ", "",  file=outfile, sep="\n")

for (j in 1:replicates){
	times <- list()

	for (i in 1:length(dat[,1])){
		if (dat$min_age[i] == 0){
			dat$min_age[i] <- dat$min_age[i] + 0.001
		}
	}

	if (isTRUE(rnd)){
			dat$new_age <- round(runif(length(dat[,1]), min=dat[,3], max=dat[,4]), digits=6)
		} else {
			for (i in 1:length(dat[,1])){
				dat$new_age[i] <- mean(c(dat[i,3], dat[i,4]))
			}				
		}

	dat2 <- subset(dat, select=c("Species","new_age"))
	taxa <- sort(unique(dat2$Species))

	for (n in 1:length(taxa)){
		times[[n]] <- dat2$new_age[dat2$Species == taxa[n]]
		if (splist$Status[splist$Species == taxa[n]] == "extant"){
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

splistout <- paste(dirname(file), "/", fname, "_SpeciesList.txt", sep="")
lookup <- as.data.frame(taxa)
lookup$status  <- "extinct"

write.table(splist, file=splistout, sep="\t", row.names=F, quote=F)
cat("  PyRate input file was saved in: ", sprintf("%s", outfile), "\n\n")

}


fit.prior <- function(file = NULL, lineage = "root_age"){

require(fitdistrplus)

if (is.null(file)){
    stop("You must enter a valid filename.\n")
	}

dat <- read.table(file, header=T, stringsAsFactors=F, row.names=NULL)
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
