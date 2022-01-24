# Dispersal Extinction Sampling models
#### January 2022
***
#### Contents
* [Preparing input data for DES analysis](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md#input-data-preparation)
* [Running a DES analysis](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md#running-a-des-analysis)

* [Return to Index](https://github.com/dsilvestro/PyRate/tree/master/tutorials#pyrate-tutorials---index) 
***


## Input data preparation
The DES model needs a set of replicated input files with the taxon occurrences classified into time-bins and areas. These can easily be derived from two tab-delimited text files or with the R package [speciesgeocodeR](https://github.com/azizka/speciesgeocodeR). The data needed are: 1) a table with the fossil occurrences 2) a table with the recent distribution of all taxa. The distributions can be provide either as discrete areas classification (Example 1) or as occurrence coordinates (Example 2). Replication arises from a uniform resampling of the fossil age estimates between the earliestAge and latestAge of each fossil.

#### Example 1 - Discrete area classification input

In the first example the taxon distribution is already available in the areas of interest for the DES analysis (Eurasia and North America). Example input and output files can be found [here](https://github.com/dsilvestro/PyRate/tree/master/example_files/DES_examples/Carnivora). The format of the input data follows the DarwinCore standard. [CarnivoraFossils.txt](https://github.com/dsilvestro/PyRate/blob/master/example_files/DES_examples/Carnivora/CarnivoraFossils.txt) shows the minimum input data for fossil occurrences.

| scientificName   | earliestAge  | latestAge | higherGeography |
| ------------- |:-------------:| -----:| -----:|
Acheronictis | 30.8 | 20.43 | NAmerica
Adcrocuta | 11.608 | 5.333 | Eurasia
Agriotherium | 10.3 | 4.9 | NAmerica
Agriotherium | 11.608 | 5.333 | Eurasia
Bassariscus | 20.43 | 15.97 | NAmerica
Bassariscus | 4.9 | 1.8 | NAmerica
Canis | 4.9 | 0.3 | NAmerica
Canis | 1.8 | 0.3 | NAmerica
Canis | 3.2 | 0.0117 | Eurasia

[CarnivoraRecent.txt](https://github.com/dsilvestro/PyRate/blob/master/example_files/DES_examples/Carnivora/CarnivoraRecent.txt) shows the recent distribution. Note that, if a taxon occurs in both areas in recent time, it needs a separate row for each area.

| scientificName   | higherGeography |
| ------------- |:-------------:|
Acinonyx | Eurasia
Bassaricyon | NAmerica
Bassariscus | NAmerica
Canis | Eurasia
Canis | NAmerica

To launch PyRateDES open a Terminal window and browse to the PyRate directory 

`cd /path/to/PyRate`

The following code produces the DES input files.

`python ./PyRateDES2.py -fossil .../example_files/DES_examples/Carnivora/CarnivoraFossils.txt -recent .../example_files/DES_examples/Carnivora/CarnivoraRecent.txt -wd .../example_files/DES_examples/Carnivora -filename Carnivora -bin_size 0.5 -rep 10`

* `-fossil` is the path to the table with fossil occurrences.

* `-recent` is the path to the table with the recent distribution.

* `-wd` is the path to the generated inpute file(s).

* `-filename` is the name of the generated inpute file(s) for the DES analysis.

* `-bin_size` defines the size of the time bins (in million years if fossil ages are in million years). Bin size should be chosen as a compromise between the desired resolution and data availability.

* `-rep` is the number of replicates desired. Replication arises from the age uncertainty of the fossils, which are usually dated with a minimum and maximum age. For each replicate the age of the fossils is sampled from a uniform distribution between minimum and maximum age.

* `-trim_age` is an optional argument to truncate the dataset by an maximum age (e.g. `-trim_age 23.03`. truncates to the Neogene). It omits a fossil when the uniform resampling of the fossil age estimates exceeds the specified age.

* `-data_in_area 1` is an argument to code fossil occurrences for a DES analysis where lineages are only known from a single area.

* `-plot_raw` is an optional argument to generate a plot in PDF format in `-wd` of the observed diversity trajectories and their 95% credible interval in the two area. This requires that R is installed on your PC to execute the shell command Rscript. If you are using Windows, please make sure that the path to Rscript.exe is included in the PATH environment variables (default in Mac/Linux).

![Example observed diversity](https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/DES_observed_diversity.png)


#### Example 2 - Species distributions as coordinates

Often species distributions are available as coordinates rather than as discrete areas.  Coordinates for the fossil and recent data can be prepared for DES input with the following script using speciesgeocodeR (>= v 1.0-6). Note however, that DES assumes the present data to be complete, so a use of occurrence records for recent data is only for very well samples and curated data. Additionally usually native species ranges should be used (avoiding the anthropogenic influence, which will inflate, or sometimes decrease, the recent distribution of many taxa).

```{r, warning = F, echo = F}
library(speciesgeocodeR)

occ.thresh <- 0.1 #at least 10% occurrence in an area required

#Assign the fossil coordinates to operational areas
fos <- read.table(".../PyRate/example_files/DES_examples/DES_input_data/Example_2_coordinates.txt", sep = "\t", header = TRUE)
fos.class <- SpGeoCod(".../PyRate/example_files/DES_examples/DES_input_data/Example_2_coordinates.txt",
                      ".../PyRate/example_files/DES_examples/DES_input_data/Example_regions.shp", areanames = "Region")
foss <- data.frame(fos, higherGeography = fos.class$sample_table$homepolygon)
foss <- foss[complete.cases(foss),]

#Assign the recent coordinates to operational areas, using the occurrence threshold
rec <- read.table("Example_data/Example_2_recent_coordinates.txt", sep = "\t", header = TRUE)

rec.class <- SpGeoCod(".../PyRate/example_files/DES_examples/DES_input_dataExample_2_recent_coordinates.txt",
                      ".../PyRate/example_files/DES_examples/DES_input_data/Example_regions.shp", 
                      areanames = "Region")

pres <- round(rec.class$spec_table[, 2:3] / rowSums(rec.class$spec_table[, 2:3]), 2)
pres[which(pres[, 1] >= occ.thresh ), 1] <- names(pres)[1] 
pres[which(pres[, 2] >= occ.thresh ), 2] <- names(pres)[2] 
pres <- data.frame(scientificName = rep(rec.class$spec_table[, 1], 2),
                   higherGeography = c(pres[, 1], pres[, 2]))
pres <- pres[pres$higherGeography %in% names(rec.class$spec_table), ]

# Write tables 
write.table(foss, ".../PyRate/example_files/DES_examples/DES_input_data/foss.txt",
            sep = "\t", col.names = TRUE, row.names = FALSE, quote = FALSE)
write.table(pres, ".../PyRate/example_files/DES_examples/DES_input_data/pres.txt",
            sep = "\t", col.names = TRUE, row.names = FALSE, quote = FALSE)
```

Launch PyRateDES by opening a Terminal window and browsing to the PyRate directory 

`cd /path/to/PyRate`

The following code produces the DES input files (see Example 1 for the explanation of the arguments).

`python ./PyRateDES2.py -fossil .../example_files/DES_examples/DES_input_data/foss.txt -recent .../example_files/DES_examples/DES_input_data/pres.txt -wd .../example_files/DES_examples -filename Example2 -bin_size 2 -rep 5`


## Running a DES analysis
To launch PyRateDES open a Terminal window and browse to the PyRate directory 

`cd /path/to/pyrate/code`

PyRateDES2.py is an upgraded version of the original DES model which allows more flexibility in time-variable dispersal and extinction models.

#### Time variable model (rate shifts)
Launch a maximum likelihood analysis with shifts in preservation, dispersal, and extinction rates at 5.3 and 2.6 Ma.

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -TdD -TdE`

The `-TdD` and `-TdE` commands specify time-dependent dispersal and extinction rates, respectively.

The command `-A 2` specifies that you want to use a maximum likelihood algorithm. 

Note that if `input_data.txt` is in not in he same directory as `PyRateDES2.py` you need to specify its full path. The output files will be saved where the `input_data.txt` was.

#### Covariate D/E models
You can use a time variable predictor (e.g. sea level or temperature) and model dispersal and/or extinction as an exponential function of the predictor. 

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -varD predictor_file.txt -varE predictor_file.txt`

Note that if `predictor_file.txt ` is in not in he same directory as `PyRateDES2.py` you need to specify its full path. The format of the predictor file is a tab-separated text file with column headers as shown in PyRate's repository (`/PhanerozoicTempSmooth.txt`).

#### Mixed D/E models
You can use different predictors for dispersal and extinction, e.g.

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -varD predictor_file1.txt -varE predictor_file2.txt`

For instance you can test sea level as a predictor of dispersal and a climate proxy as a predictor for extinction. 

#### DES analysis with heterogeneity in preservation rates across taxa
You can include differences in preservation rates across taxa. The command `-mG` specifies a model where the mean preservation rate across all taxa equals q and the heterogeneity is given by a discretized Gamma distribution with `-ncat` categories.

`./PyRateDES2.py -d input_data.txt -A 2 -mG -ncat 4 -TdD -TdE`

##### The likelihoods of different models (with different predictors or rate shifts) can be compared to perform model testing, for example using AIC scores. 















