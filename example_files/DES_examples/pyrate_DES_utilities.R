# # DES in function
# DESin <- function(x, recent, bin.size, reps = 1, verbose = T) {
#
#   # load data
#   if (is.data.frame(x)) {
#     dat <- x
#   }else{
#     dat <- read.table(x, sep = "\t", header = T, row.names = NULL)
#   }
#
#   nes <- c("scientificName", "earliestAge", "latestAge", "higherGeography")
#   if(!all(nes %in% names(dat))){
#     stop(sprintf("did not find column %s. Check input data", nes[!nes %in% names(dat)]))
#   }
#
#   if(! "midpointAge" %in% names(dat)){
#     dat$midpointAge <- (dat$earliestAge + dat$latestAge)/2
#     #warning("column midpointAge not found, calculating from earliestAge and latestAge")
#   }
#
#
#   # load and prepare recent data
#   if (is.data.frame(recent)) {
#     rece <- recent
#   }else{
#     rece <- read.table(recent, header = T, sep = "\t", stringsAsFactors = F, row.names = NULL)
#   }
#
#   nes <- c("scientificName", "higherGeography")
#   if(!all(nes %in% names(rece))){
#     stop(sprintf("did not find column %s. Check input data", nes[!nes %in% names(rece)]))
#   }
#
#   rece$higherGeography <- as.character(rece$higherGeography)
#   rece[rece$higherGeography == sort(unique(rece$higherGeography))[1], "higherGeography"] <- 1
#   rece[rece$higherGeography == sort(unique(rece$higherGeography))[2], "higherGeography"] <- 2
#   rece <- unique(rece)
#   rece$higherGeography <- as.numeric(rece$higherGeography)
#   rece <- aggregate(higherGeography ~ scientificName, data = rece, sum)
#
#   # code fossil data
#   outp <- list()
#   for (i in 1:reps) {
#     if (verbose == TRUE) {
#       print(sprintf("producing replicate %s of %s", i, reps))
#     }
#
#     # simulate random age between min and max
#     dat$age <- sapply(seq(1, length(dat$scientificName)), function(x) runif(1, max = dat$earliestAge[x],
#                                                                    min = dat$latestAge[x]))
#
#     # define age class cutter and cut ages into timebins
#     cutter <- seq(0, max(dat$age), bin.size)
#     dat$timeint <- as.numeric(as.character(cut(dat$age, breaks = cutter,
#                                                digits = 5, labels = cutter[-length(cutter)])))
#
#     # code the presence in each regions per scientificName
#     dat.list <- split(dat, dat$scientificName)
#     binned <- lapply(dat.list, function(x) {
#       dat.out <- data.frame(timebin = cutter, higherGeography1 = rep(0, length(cutter)),
#                             higherGeography2 = rep(0, length(cutter)))
#       if (length(x$higherGeography == sort(unique(dat$higherGeography))[1]) > 0) {
#         dat.out[dat.out$timebin %in% x[x$higherGeography == sort(unique(dat$higherGeography))[1],
#                                        "timeint"], "higherGeography1"] <- 1
#       }
#       if (length(x$higherGeography == sort(unique(dat$higherGeography))[2]) > 0) {
#         dat.out[dat.out$timebin %in% x[x$higherGeography == sort(unique(dat$higherGeography))[2],
#                                        "timeint"], "higherGeography2"] <- 2
#       }
#       presence <- rowSums(dat.out[, 2:3])
#       return(presence)
#     })
#
#     # set timebins before first appearance to NaN
#     out <- lapply(binned, function(x) {
#       if (length(which(x > 0)) == 0) {
#         x <- rep("nan", length(x))
#         return(as.numeric(x))
#       } else {
#         if (max(which(x > 0)) < length(x)) {
#           x[(max(which(x > 0)) + 1):length(x)] <- "nan"
#           return(as.numeric(x))
#         } else {
#           return(x)
#         }
#       }
#     })
#
#     # output format
#     out <- do.call("rbind.data.frame", out)
#     names(out) <- (cutter + bin.size/2)
#     # out <- out[,as.numeric(names(out)) < age.cut]
#     out <- rev(out)
#     outp[[i]] <- out
#   }
#
#   # combine recent and fossil data
#   outp2 <- lapply(outp, function(x) {
#     outo <- merge(x, rece, by.x = "row.names", by.y = "scientificName", all.x = T)
#     outo$higherGeography[is.na(outo$higherGeography)] <- 0
#     names(outo)[1] <- "scientificName"
#     names(outo)[ncol(outo)] <- 0
#     # outo2 <- data.frame(apply(outo, 2, function(x) as.character(x)), stringsAsFactors = F)
#      # names(outo2) <- names(outo)
#     return(outo)
#   })
#
#   outp <- list(dat, rece, outp2, bin.size)
#   class(outp) <- "DES.in"
#   return(outp)
# }
#
# # write to working directory
# write.DES.in <- function(x, file) {
#   for (i in 1:length(x[[3]])) {
#     write.table(x[[3]][[i]], paste(file, "_rep", i, ".txt", sep = ""), na = "NaN",
#                 sep = "\t", row.names = F, quote = F)
#   }
# }
#
# plot.DES.in <- function(x, plottype = c("all", "samplelocations", "inputviz", "replicates"),
#                         xlim = c(-180, 180), ylim = c(-90, 90), pch = 1, ...) {
#   match.arg(plottype)
#
#   if(!all(c("decimalLongitude", "decimalLatitude") %in% names(x[[1]]))){
#     warning("no coordinates found, no locations are plotted")
#   }
#
#   if ("all" %in% plottype) {
#     .SampleLocations(x, ...)
#     .InputData(x, ...)
#     .ReplicateAges(x)
#   } else {
#
#     if ("samplelocations" %in% plottype) {
#       .SampleLocations(x, ...)
#     }
#     if ("occurrencetimes" %in% plottype) {
#       .IputData(x, ...)
#     }
#     if ("inputviz" %in% plottype) {
#       .InputData(x, ...)
#     }
#     if ("replicates" %in% plottype) {
#       .ReplicateAges(x)
#     }
#   }
# }
#
# # summary
# summary.DES.in <- function(x) {
#   ares <- split(x[[1]], f = x[[1]]$higherGeography)
#
#   list(Number_of_areas = length(ares),
#        Data = data.frame(row.names = c("Timerange_min", "Timerange_max", "Number of records",
#                                        "Mean record age", "Number of taxa",  "Mean taxon age"),
#                          Area_1 = c(min(ares[[1]]$midpointAge), max(ares[[1]]$midpointAge),
#                                     nrow(ares[[1]]), round(mean(ares[[1]]$midpointAge), 1),
#                                     length(unique(ares[[1]]$scientificName)),
#                                     round(mean(aggregate(ares[[1]]$midpointAge, by = list(ares[[1]]$scientificName),min)$x), 1)),
#                          Area_2 = c(min(ares[[2]]$midpointAge), max(ares[[2]]$midpointAge),
#                                     nrow(ares[[2]]), round(mean(ares[[2]]$midpointAge), 1), length(unique(ares[[2]]$scientificName)),
#                                     round(mean(aggregate(ares[[2]]$midpointAge, by = list(ares[[2]]$scientificName), min)$x), 1))),
#        Number_of_Replicates = length(x[[3]]))
# }
#
# #auxillary functions
# .SampleLocations <- function(x, ...) {
#   map("world", ...)
#   points(x[[1]]$decimalLongitude, x[[1]]$decimalLatitude, col = x[[1]]$higherGeography, ...)
#   axis(1)
#   axis(2)
#   title("Input fossil locations")
#   legend("topright", legend = unique(x[[1]]$higherGeography), fill = c("blue", "red"))
#   box("plot")
# }
#
# .InputData <- function(x, ...) {
#
#   # Number of samples
#   occ.all <- table(x[[1]]$midpointAge)
#   occ.reg1 <- table(subset(x[[1]], x[[1]]$higherGeography == unique(x[[1]]$higherGeography)[1],
#                            select = "midpointAge"))
#   occ.reg2 <- table(subset(x[[1]], x[[1]]$higherGeography == unique(x[[1]]$higherGeography)[2],
#                            select = "midpointAge"))
#   plot(1, 1, xlim = c(max(as.numeric(names(occ.reg1))), min(as.numeric(names(occ.reg1)))),
#        ylim = c(min(occ.all), max(occ.all)), xlab = "Time",
#        ylab = "Number of Records", type = "n")
#   points(occ.reg1 ~ as.numeric(names(occ.reg1)), type = "b", col = "blue", ...)
#   points(occ.reg2 ~ as.numeric(names(occ.reg2)), type = "b", col = "red", ...)
#   legend("topleft", legend = as.character(unique(x[[1]]$higherGeography)),
#          col = c("blue", "red"), lty = 1, pch = 1)
#   title("Number of samples")
#
#   # Number of taxa
#   dd <- split(x[[1]], f = x[[1]]$higherGeography)
#   dd <- lapply(dd, function(x) aggregate(x$scientificName, by = list(x$midpointAge,
#                                                             x$scientificName), length))
#   dd <- lapply(dd, function(x) aggregate(x$Group.2, by = list(x$Group.1),
#                                          length))
#
#   plot(1, 1, xlim = c(max(c(as.numeric(dd[[1]]$Group.1)), as.numeric(dd[[2]]$Group.1)),
#                       min(c(as.numeric(dd[[1]]$Group.1)), as.numeric(dd[[2]]$Group.1))),
#        ylim = c(min(c(dd[[1]]$x), dd[[2]]$x), max(c(dd[[1]]$x), dd[[2]]$x)),
#        xlab = "Time", ylab = "Number of Records",
#        type = "n")
#
#   points(dd[[1]]$x ~ as.numeric(dd[[1]]$Group.1), type = "b", col = "blue")  #, ...)
#   points(dd[[2]]$x ~ as.numeric(dd[[2]]$Group.1), type = "b", col = "red")  #, ...)
#   legend("topleft", legend = as.character(unique(x[[1]]$higherGeography)),
#          col = c("blue", "red"), lty = 1, pch = 1)
#   title("Number of taxa")
#
#   # boxplot fossil ages
#   boxplot(x[[1]]$midpointAge ~ x[[1]]$higherGeography, col = c("blue", "red"))
#   title("Fossil ages")
#
#   # Number of records per taxon
#   tax.num <- aggregate(x[[1]]$midpointAge, by = list(x[[1]]$scientificName, x[[1]]$higherGeography),
#                        length)
#   boxplot(tax.num$x ~ tax.num$Group.2, col = c("blue", "red"))
#   title("Number of records per Taxon")
#
#   #fraction of taxa per area in recent data
#   frq <- round((table(x[[2]]$higherGeography)/ sum(table(x[[2]]$higherGeography))), 2)
#   barplot(frq)
#   box("plot")
#   title("Fraction of Taxa per Area (present day)")
# }
#
# .ReplicateAges <- function(x) {
#   meas <- unlist(lapply(x[[3]], length))
#
#   if(isTRUE(all.equal(max(meas) , min(meas)))){
#     dat <-x[[3]]
#   }else{
#     numb <- which(meas < max(meas))
#     for(i in 1:length(numb)){
#       dat <- x[[3]]
#       dat[[numb[i]]] <- c(rep(0, length(numb)), dat[[numb[i]]])
#       names(dat[[numb[i]]])[1] <- (max(as.numeric(names(dat[[numb[i]]])), na.rm = T) + x[[4]])
#     }
#
#   }
#
#   reg1 <- lapply(x[[3]], function(k) {apply(k[,-1], 2, function(z){
#     test <- z[as.numeric(z) == 1]
#     test <- length(test[complete.cases(test)])
#     return(test)})})
#   are1all <- do.call("rbind.data.frame", reg1)
#   names(are1all) <- names(reg1[[1]])
#
#   reg2 <- lapply(x[[3]], function(k) {apply(k[,-1], 2, function(z){
#     test <- z[as.numeric(z) == 2]
#     test <- length(test[complete.cases(test)])
#     return(test)})})
#   are2all <- do.call("rbind.data.frame", reg2)
#   names(are2all) <- names(reg2[[1]])
#
#   reg3 <- lapply(x[[3]], function(k) {apply(k[,-1], 2, function(z){
#     test <- z[as.numeric(z) == 3]
#     test <- length(test[complete.cases(test)])
#     return(test)})})
#   are3all <- do.call("rbind.data.frame", reg3)
#   names(are3all) <- names(reg3[[1]])
#
#   tot <- lapply(x[[3]], function(k) {apply(k[,-1], 2, function(z){
#     test <- z[as.numeric(z) %in% c(1,2,3)]
#     test <- length(test[complete.cases(test)])
#     return(test)})})
#   totall <- do.call("rbind.data.frame", tot)
#   names(totall) <- names(tot[[1]])
#
#   are1.max <- apply(are1all, 2, max)
#   are1.min <- apply(are1all, 2, min)
#
#   are2.max <- apply(are2all, 2, max)
#   are2.min <- apply(are2all, 2, min)
#
#   plot(1, 1, xlim = c(max(c(as.numeric(names(are1)), as.numeric(names(are2)))),
#                       min(c(as.numeric(names(are1)), as.numeric(names(are2))))),
#        ylim = c(min(c(are1.min, are2.min)), max(c(are1.max, are2.max))),
#        xlab = "Time", ylab = "Number of Taxa",
#        type = "n")
#   polygon(c(names(are1), rev(names(are1))), c(are1.min, rev(are1.max)),
#           col = rgb(0, 0, 255, 125, maxColorValue = 255),
#           border = rgb(0, 0, 255, 125, maxColorValue = 255))
#   polygon(c(names(are1), rev(names(are2))), c(are2.min, rev(are2.max)),
#           col = rgb(255, 0, 0, 125, maxColorValue = 255),
#           border = rgb(255, 0, 0, 125, maxColorValue = 255))
#   legend("topleft", legend = unique(x[[1]]$higherGeography),
#          fill = c(rgb(0, 0, 255, 125, maxColorValue = 255), rgb(255, 0, 0, 125, maxColorValue = 255)))
#   title(sprintf("Taxon number randomized ages, replicates = %s",length(x[[3]])))
#
#
#   #Faction per area through time
#   reg1all <- round(are1all/totall, 2)
#   reg1.min <- apply(reg1all, 2, min)
#   reg1.max <- apply(reg1all, 2, max)
#
#   reg2all <- round(are2all/totall, 2)
#   reg2.min <- apply(reg2all, 2, min)
#   reg2.max <- apply(reg2all, 2, max)
#
#   reg3all <- round(are3all/totall, 2)
#   reg3.min <- apply(reg3all, 2, min)
#   reg3.max <- apply(reg3all, 2, max)
#
#   # plot
#   plot(1, 1, xlim = c(max(as.numeric(names(reg1all))),
#                       min(as.numeric(names(reg1all)))),
#        ylim = c(0, 1.1),
#        xlab = "Time", ylab = "Fraction of Taxa",
#        type = "n")
#   polygon(c(names(reg1all), rev(names(reg1all))), c(reg1.min, rev(reg1.max)),
#           col = rgb(0, 0, 255, 100, maxColorValue = 255),
#           border = rgb(0, 0, 255, 100, maxColorValue = 255))
#   polygon(c(names(reg2all), rev(names(reg2all))), c(reg2.min, rev(reg2.max)),
#           col = rgb(255, 0, 0, 100, maxColorValue = 255),
#           border = rgb(255, 0, 0, 100, maxColorValue = 255))
#   polygon(c(names(reg3all), rev(names(reg3all))), c(reg3.min, rev(reg3.max)),
#           col = rgb(0, 255, 0, 100, maxColorValue = 255),
#           border = rgb(0, 255, 0, 100, maxColorValue = 255))
#   legend("top", legend = c(as.character(unique(x[[1]]$higherGeography)), "both"),
#          fill = c(rgb(0, 0, 255, 125, maxColorValue = 255),
#                   rgb(255, 0, 0, 125, maxColorValue = 255),
#                   rgb(0, 255, 0, 125, maxColorValue = 255)), ncol = 3)
#   title(sprintf("Fraction of taxa per area, replicates = %s",length(x[[3]])))
# }
#
#
#
DESin <- function(x, 
                  recent, 
                  taxon = "scientificName",
                  area = "higherGeography",
                  age1 = "earliestAge",
                  age2 = "latestAge",
                  bin.size = 5, 
                  reps = 3, 
                  verbose = FALSE) {
    
    # load data
    if (is.data.frame(x)) {
        dat <- x
    } else {
        dat <- read.table(x, sep = "\t", header = TRUE, row.names = NULL)
    }

    # CHECK IF this is still necessary, and why the summary method still uses
    # midpoints
    
    if (!"midpointage" %in% names(dat)) {
        dat$midpointage <- (dat[[age1]] + dat[[age2]])/2
        warning("column midpointage not found, calculating from earliestage and latestage")
    }
    
    # load and prepare recent data
    if (is.data.frame(recent)) {
        rece <- recent
    }else {
        rece <- read.table(recent, header = TRUE, sep = "\t", stringsAsFactors = FALSE, 
            row.names = NULL)
    }
    

    rece[[area]] <- as.character(rece[[area]])
    rece[rece[[area]] == sort(unique(rece[[area]]))[1], area] <- 1
    rece[rece[[area]] == sort(unique(rece[[area]]))[2], area] <- 2
    rece <- unique(rece)
    rece[[area]] <- as.numeric(rece[[area]])
    rece <- aggregate(rece[[area]] ~ rece[[taxon]], FUN = sum)
    names(rece) <- c(taxon, area)

    # code fossil data
    outp <- list()
    for (i in 1:reps) {
        if (verbose) {
            print(sprintf("producing replicate %s of %s", i, reps))
        }
        
        # simulate random age between min and max
        dat$age <- sapply(seq(1, nrow(dat)), function(x) stats::runif(1, max = dat[[age1]][x], 
            min = dat[[age2]][x]))
        
        # define age class cutter and cut ages into timebins
        cutter <- seq(0, max(dat$age) + bin.size, by = bin.size)
        dat$timeint <- as.numeric(as.character(cut(dat$age, breaks = cutter, 
            digits = 5, labels = cutter[-length(cutter)])))
        
        # code the presence in each regions per species
        dat.list <- split(dat, dat[[taxon]])
        
        binned <- lapply(dat.list, function(k) {
            dat.out <- data.frame(timebin = cutter, 
                                  area1 = rep(0, length(cutter)), 
                                  area2 = rep(0, length(cutter)))
            
            if (length(k[[area]] == sort(unique(dat[[area]])[1])) > 0) {
                dat.out[dat.out$timebin %in% unlist(k[k[[area]] == unique(dat[[area]])[1], "timeint"]), "area1"] <- 1
            }
            if (length(k[[area]] == sort(unique(dat[[area]])[2])) > 0) {
                dat.out[dat.out$timebin %in% unlist(k[k[[area]] == unique(dat[[area]])[2], "timeint"]), "area2"] <- 2
            }
            presence <- rowSums(dat.out[, 2:3])
            return(presence)
        })
        
        # set timebins before first appearance to NaN
        out <- lapply(binned, function(k) {
            if (!any(k > 0)) {
                k <- rep("nan", length(k))
                return(as.numeric(k))
            } else {
                if (max(which(k > 0)) < length(k)) {
                  k[(max(which(k > 0)) + 1):length(k)] <- "nan"
                  return(as.numeric(k))
                } else {
                  return(k)
                }
            }
        })
        
        # output format
        out <- do.call("rbind.data.frame", out)
        names(out) <- (cutter + bin.size / 2)
        out <- rev(out)
        out[[taxon]] <- names(dat.list)
        outp[[i]] <- out
    }
    
    
    # combine recent and fossil data
    outp2 <- lapply(outp, function(k) {
        outo <- merge(k, rece, by = taxon, all.x = TRUE)
        outo[[area]][is.na(outo[[area]])] <- 0
        #rownames(outo) <- outo[, 1]
        #outo <- outo[, -1]
        names(outo)[ncol(outo)] <- 0
        return(outo)
    })
    
    # make sure all replicates cover the same time spann, i.e. add additional
    # columns before the first time column
    meas <- sapply(outp2, "ncol")
    
    if (max(meas) != min(meas)) {
        numb <- which(meas < max(meas))
        for (i in numb) {
            dat.int <- outp2[[i]]
            repl <- nrow(dat.int) * (max(meas) - meas[i])  # how many NaNs are needed
            dat.comb <- c(rep(NaN, times = repl), unlist(dat.int[,-1]))
            dat.int <- data.frame(matrix(dat.comb, 
                                         nrow = nrow(dat.int), 
                                         ncol = max(meas)-1, 
                byrow = FALSE))
            dat.int <- data.frame(outp[[i]][taxon],
                                  dat.int)
            names(dat.int) <- names(outp2[[which(meas == max(meas))[1]]])
            #rownames(dat.int) <- rownames(outp2[[i]])
            outp2[[i]] <- dat.int
        }
    }
    
    # create output object
    outp <- list(input_fossils = dat, 
                 input_recent = rece, 
                 DES_replicates = outp2,
                 bin_size = bin.size,
                 area = area,
                 taxon = taxon)
    names(outp) <- c("input_fossils", "input_recent", "DES_replicates", "bin_size", "area", "taxon")
    class(outp) <- c("DESin", "list")
    return(outp)
}

is.DESin <- function(x) {
    inherits(x, "DESin")
}

plot.DESin <- function(x, ribbon = TRUE, ...) {
    # species in all areas
    area1 <- lapply(x[["DES_replicates"]], function(k) {
        k[is.na(k)] <- 0
        k[k != 1] <- 0
        colSums(k[,-1])
    })
    area1 <- do.call("rbind.data.frame", area1)
    
    area2 <- lapply(x[["DES_replicates"]], function(k) {
        k[is.na(k)] <- 0
        k[k != 2] <- 0
        k[k == 2] <- 1
        colSums(k[,-1])
    })
    area2 <- do.call("rbind.data.frame", area2)
    
    areaB <- lapply(x[["DES_replicates"]], function(k) {
        k[is.na(k)] <- 0
        k[k != 3] <- 0
        k[k == 3] <- 1
        colSums(k[,-1])
    })
    areaB <- do.call("rbind.data.frame", areaB)
    
    times <- as.numeric(as.character(names(x[["DES_replicates"]][[1]][-1])))
    
    dat.plo <- data.frame(time = rep(times, 3), 
                          mean = c(round(colMeans(area1), 1),
                                   round(colMeans(area2), 1), 
                                   round(colMeans(areaB), 1)), 
                          lwr = c(do.call(pmin, data.frame(t(area1))), 
                                  do.call(pmin, data.frame(t(area2))),
                                  do.call(pmin, data.frame(t(areaB)))), 
                          upr = c(do.call(pmax, data.frame(t(area1))), 
                                  do.call(pmax, data.frame(t(area2))),
                                  do.call(pmax, data.frame(t(areaB)))), 
                          area = c(rep("Area1", length(times)), 
                                   rep("Area2", length(times)), 
                                   rep("Both", length(times))))
    
    plo <- ggplot() + 
      geom_line(data = dat.plo, 
                aes_string(x = "time", y = "mean",
                           group = "area", col = " area")) +
      scale_x_reverse() + 
      xlab("Time") + 
      ylab("Species") + 
      theme_bw() + 
      theme(legend.title = element_blank())
    
    if (ribbon) {
        plo <- plo + 
          geom_ribbon(data = dat.plo, 
                      aes_string(x = "time", 
                                 ymax = "upr",
                                 ymin = "lwr", 
                                 group = "area",
                                 fill = "area"), 
                      alpha = 1/5)
    }
    plo
}

summary.DESin <- function(object, ...) {
    ares <- split(object[["input_fossils"]], f = object[["input_fossils"]][[object$area]])
    
    outp.nams <- c("Minimum_age", 
                   "Maximum_age", 
                   "Number of records",
                   "Mean record age", 
                   "Number of taxa",
                   "Mean taxon age")
    
    area.1 <- c(round(min(ares[[1]]$midpointage), 1),
                round(max(ares[[1]]$midpointage),1),
                nrow(ares[[1]]), 
                round(mean(ares[[1]]$midpointage), 1), 
                length(unique(ares[[1]][[object$taxon]])),
                round(mean(aggregate(ares[[1]]$midpointage, by = list(ares[[1]][[object$taxon]]),min)$x), 1))
    
    area.2 <- c(round(min(ares[[2]]$midpointage), 1),
                round(max(ares[[2]]$midpointage),  1), 
                nrow(ares[[2]]),
                round(mean(ares[[2]]$midpointage), 1),
                length(unique(ares[[2]][[object$taxon]])),
                round(mean(aggregate(ares[[2]]$midpointage, by = list(ares[[2]][[object$taxon]]), min)$x), 1))
    
    list(Number_of_areas = length(ares), 
         Input_Data = data.frame(row.names = outp.nams, 
                                 Area_1 = area.1, 
                                 Area_2 = area.2), 
         Number_of_Replicates = length(object[["DES_replicates"]]))
}

write.DESin <- function(x, file) {
    for (i in 1:length(x[["DES_replicates"]])) {
        write.table(x[["DES_replicates"]][[i]], paste(file, "_rep", i, ".txt", 
            sep = ""), na = "NaN", sep = "\t", row.names = F, quote = FALSE)
    }
}