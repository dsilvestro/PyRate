#by Alexander Zizka, 2016-01-30, bugs & questions to speciesgeocodeR@googlegroups.com

setwd("C:/Users/xzizal/Desktop/GitHub/PyRate")

library(maps)
source("pyrate_DES_utilities.R")

#Example 1 - Minimum data

#this generates an DESin R object; 
#Presence data = present distribution of all recent taxa also in the fossil dataset
#bin.size = The size of the time bins, here 2 million years
#reps = the number of replicates for the 

exp1 <- DESin("Example_1_minimum_data.txt", "Example_1_recent_distributions.txt", 
              bin.size = 2, reps = 3)

#Write the input files for PyRates/the DES model; this will crete one file per replicate
write.DES.in(exp1, file = "Example1_DES_in")

#Explore data for quality control and bias estimation
summary(exp1)

par(ask = T)
plot(exp1)


#####Example 2 - Starting with coordinates
#this needs speciesgeocodeR 1.0-5, download from https://github.com/azizka/speciesgeocodeR
library(speciesgeocodeR)

occ.thresh <- 0.1 #at least 10% occurrence in an area required

#Assign the fossil coordinates to operational areas
fos <- read.table("example_files/DES_input_data/Example_2_coordinates.txt", sep = "\t", header = T)
fos.class <- SpGeoCod("example_files/DES_input_data/Example_2_coordinates.txt", "example_files/DES_input_data/Example_regions.shp", areanames = "Region")
foss <- data.frame(fos, higherGeography = fos.class$sample_table$homepolygon)
foss <- foss[complete.cases(foss),]

#Assign the recent coordinates to operational areas, using the occurrence threshold
rec <- read.table("example_files/DES_input_data/Example_2_recent_coordinates.txt", sep = "\t", header = T)

rec.class <- SpGeoCod("example_files/DES_input_data/Example_2_recent_coordinates.txt", "example_files/DES_input_data/Example_regions.shp", 
                      areanames = "Region")

pres <- round(rec.class$spec_table[, 2:3] / rowSums(rec.class$spec_table[, 2:3]), 2)
pres[which(pres[, 1] >= occ.thresh ), 1] <- names(pres)[1] 
pres[which(pres[, 2] >= occ.thresh ), 2] <- names(pres)[2] 
pres <- data.frame(scientificName = rep(rec.class$spec_table[, 1], 2),
                   higherGeography = c(pres[, 1], pres[, 2]))
pres <- pres[pres$higherGeography %in% names(rec.class$spec_table), ]

#create DESin files
exp2 <- DESin(foss, pres, bin.size = 2, reps = 3)
write.DES.in(exp2, "Example2_out")

#explore data
summary(exp2)

par(ask = T)
plot(exp2)
