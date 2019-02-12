# Dispersal Extinction Sampling models
#### Feb 2019
***
#### Contents
* [Preparing input data for DES analysis](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md#input-data-preparation)
* [Running a DES analysis (draft tutorial)](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md#running-a-des-analysis)

* [Return to Index](https://github.com/dsilvestro/PyRate/tree/master/tutorials#pyrate-tutorials---index) 
***


## Input data preparation
The DES model needs a set of replicated input files with the taxon occurrences classified into time-bins and areas. These can easily be derived from two tab-delimited text using the R utilities provided in the PyRate home directory or with the R package [speciesgeocodeR](https://github.com/azizka/speciesgeocodeR). The data needed are: 1) a table with the fossil occurrences 2) a table with the recent distribution of all taxa. The distributions can be provide either as discrete areas classification (Example 1) or as occurrence coordinates (Example 2). Replication arises from a uniform resampling of the fossil age estimates between the earliestAge and latestAge of each fossil.

####Example 1 - Discrete area classification input

In the first example the taxon distribution is already available in the areas of interest for the DES analysis ("neartic" and "paleoartic"). Example input and output files can be found [here](https://github.com/dsilvestro/PyRate/tree/master/example_files/DES_input_data). The format of the input data follows the DarwinCore standard. [Example_1_minimum_data.txt](https://github.com/dsilvestro/PyRate/blob/master/example_files/DES_input_data/Example_1_minimum_data.txt) shows the minimum input data for fossil occurrences and [Example_1_recent_distributions.txt](https://github.com/dsilvestro/PyRate/blob/master/example_files/DES_input_data/Example_1_recent_distributions.txt) shows the recent distribution. Note that, if a taxon occurs in both areas in recent time, it needs a separate row for each area. Loading the R utilities file "pyrate_DES_utilities.R", the following code produces the DES input files (see example here). The code is also available [here](https://github.com/dsilvestro/PyRate/blob/master/pyrate_DES_input_example_script.R).

The `DESin function` converts the input data in DES format. `bin.size` defines the size of the time bins (in million years if fossil ages are in million years). Bin size should be chosen as a compromise between the desired resolution and data availability.`reps` Is the number of replicates desired. Replication arises from the age uncertainty of the fossils, which are usually dated with a minimum and maximum age. For each replicate the age of the fossils is sampled from a uniform distribution between minimum and maximum age.

```{r, warning = F, echo = F}
#load utility functions
source(DES_input_preparation.R)

#load input files and convert into DES format
exp1 <- DESin("Example_data/Example_1_minimum_data.txt", "Example_data/Example_1_recent_distributions.txt", 
              bin.size = 2, reps = 3)
 
#explore the data for potential biases
summary(exp1)
par(ask = T)
plot(exp1)

#write DES input files to the working directory
write.DES.in(exp1, file = "Example_data/Example1_DES_in")

```


####Example 2 - Species distributions as coordinates

Often species distributions are available as coordinates rather than as discrete areas.  Coordinates for the fossil and recent data can be prepared for DES input with the following script using speciesgeocodeR (>= v 1.0-6). Note however, that DES assumes the present data to be complete, so a use of occurrence records for recent data is only for very well samples and curated data. Additionally usually native species ranges should be used (avoiding the anthropogenic influence, which will inflate, or sometimes decrease, the recent distribution of many taxa).

```{r, warning = F, echo = F}
library(speciesgeocodeR)
source(DES_input_preparation.R)

occ.thresh <- 0.1 #at least 10% occurrence in an area required

#Assign the fossil coordinates to operational areas
fos <- read.table("Example_data/Example_2_coordinates.txt", sep = "\t", header = T)
fos.class <- SpGeoCod("Example_data/Example_2_coordinates.txt", "Example_data/Example_regions.shp", areanames = "Region")
foss <- data.frame(fos, higherGeography = fos.class$sample_table$homepolygon)
foss <- foss[complete.cases(foss),]

#Assign the recent coordinates to operational areas, using the occurrence threshold
rec <- read.table("Example_data/Example_2_recent_coordinates.txt", sep = "\t", header = T)

rec.class <- SpGeoCod("Example_data/Example_2_recent_coordinates.txt", "Example_data/Example_regions.shp", 
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

```


## Running a DES analysis
To launch PyRateDES open a Terminal window and browse to the PyRate directory 

`cd /path/to/pyrate/code`

PyRateDES2.py is an upgraded version of the original DES model which allows more flexibility in time-variable dispersal and extinction models.

#### Time variable model (rate shifts)
Launch a maximum likelihood analysis with shifts in preservation, dispersal, and extinction rates at 5.3 and 2.6 Ma.

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -TdD -TdE`

The `-TdD` and `-TdE` commands specify time-dependent dispersal and extinction rates, respectively.

The command `-A 2` specifies that you want to use a maximum likelihood algorithm. 

Note that if `input_data.txt` is in not in he same directory as `PyRateDES2.py` you need to specify its full path. The output files will be saved where the 


#### Covariate D/E models
You can use a time variable predictor (e.g. sea level or temperature) and model dispersal and/or extinction as an exponential function of the predictor. 

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -varD predictor_file.txt -varE predictor_file.txt`

Note that if `predictor_file.txt ` is in not in he same directory as `PyRateDES2.py` you need to specify its full path. The format of the predictor file is a tab-separated text file with column headers as shown in PyRate's repository (`/PhanerozoicTempSmooth.txt`).

#### Mixed D/E models
You can use different predictors for dispersal and extinction, e.g.

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -varD predictor_file1.txt -varE predictor_file2.txt`

For instance you can test sea level as a predictor of dispersal and a climate proxy as a predictor for extinction. 

##### The likelihoods of different models (with different predictors or rate shifts) can be compared to perform model testing, for example using AIC scores. 















