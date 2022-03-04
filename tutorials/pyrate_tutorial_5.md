# Dispersal Extinction Sampling models
#### January 2022
***
#### Contents
* [Preparing input data for DES analysis](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md#input-data-preparation)
* [Running a DES analysis](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md#running-a-des-analysis)
* [Summarizing and plotting the DES output](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md#summarizing-and-plotting-the-des-output)

* [Return to Index](https://github.com/dsilvestro/PyRate/tree/master/tutorials#pyrate-tutorials---index) 
***


## Input data preparation

The DES model needs a set of replicated input files with the taxon occurrences classified into time-bins and two discrete and predefined areas. The definition of these areas should be guided by their continuity over the study's time frame and should balance taxon occurrences and abiotic factors [(Ree & Smith, 2008)](https://academic.oup.com/sysbio/article/57/1/4/1703014). These could be biogeographic regions [(Carrillo *et al.*, 2020)](https://www.pnas.org/content/117/42/26281), insular ecosystems [(Wilke *et al.*, 2020)](https://advances.sciencemag.org/content/6/40/eabb2943), or areas without clearly defined boundaries like the tropics, whose delimitation should involve their changing extent through time [(Raja & Kiessling, 2021)](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2021.0545).

The area-coding can easily be derived from two tab-delimited text files or with the R package [speciesgeocodeR](https://github.com/azizka/speciesgeocodeR). The two text files needed are: 1) a table with the fossil occurrences 2) a table with the recent distribution of all taxa. The distributions can be provide either as discrete areas classification (Example 1) or as occurrence coordinates (Example 2). Replication arises from a uniform resampling of the fossil age estimates between the *earliestAge* and *latestAge* of each fossil.

### Example 1 - Discrete area classification input

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

`python ./PyRateDES.py -fossil .../example_files/DES_examples/Carnivora/CarnivoraFossils.txt -recent .../example_files/DES_examples/Carnivora/CarnivoraRecent.txt -wd .../example_files/DES_examples/Carnivora -filename Carnivora -bin_size 0.5 -rep 10`

* `-fossil` is the path to the table with fossil occurrences.

* `-recent` is the path to the table with the recent distribution.

* `-wd` is the path to the generated inpute file(s).

* `-filename` is the name of the generated inpute file(s) for the DES analysis.

* `-bin_size` defines the size of the time bins (in million years if fossil ages are in million years). Bin size should be chosen as a compromise between the desired resolution and data availability.

* `-rep` is the number of replicates desired. Replication arises from the age uncertainty of the fossils, which are usually dated with a minimum and maximum age. For each replicate the age of the fossils is sampled from a uniform distribution between minimum and maximum age.

* `-trim_age` is an optional argument to truncate the dataset by an maximum age (e.g. `-trim_age 23.03`. truncates to the Neogene). It omits a fossil when the uniform resampling of the fossil age estimates exceeds the specified age.

* `-taxon` is an optional argument specifying the name of the column with the taxon names in case it is not *scientificName*.

* `-area` is an optional argument specifying the name of the column with the geographic distribution in case it is not *higherGeography*.

* `-age1` is an optional argument specifying the name of the column with the earliest age in case it is not *earliestAge*.

* `-age2` is an optional argument specifying the name of the column with the latest age in case it is not *latestAge*.

* `-data_in_area 1` is an argument to code fossil occurrences for a DES analysis where lineages are only known from a single area.  For instance `python ./PyRateDES.py -fossil .../example_files/DES_examples/Diatoms_Lake_Ohrid/DiatomFossils.txt -recent .../example_files/DES_examples/Diatoms_Lake_Ohrid/DiatomRecent.txt -wd .../example_files/DES_examples/Diatoms_Lake_Ohrid -filename Diatoms -bin_size 0.5 -rep 10`

* `-plot_raw` is an optional argument to generate a plot in PDF format in `-wd` of the observed diversity trajectories and their 95% credible interval in the two area. This requires that R is installed on your PC to execute the shell command Rscript. If you are using Windows, please make sure that the path to Rscript.exe is included in the PATH environment variables (default in Mac/Linux).

![Example observed diversity](https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/DES_observed_diversity.png)


### Example 2 - Species distributions as coordinates

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

`python ./PyRateDES.py -fossil .../example_files/DES_examples/DES_input_data/foss.txt -recent .../example_files/DES_examples/DES_input_data/pres.txt -wd .../example_files/DES_examples -filename Example2 -bin_size 2 -rep 5`

<br>

### Example 3 - Simulating input data

With the R package [simDES](https://github.com/thauffe/simDES) we can simulate input files for a DES analysis under different scenarios of time-variable dispersal, extinction and sampling rates or with an influence of traits on dispersal and extinction.

```{r, warning = F, echo = F}
# Install the packages
library(devtools)
install_github("thauffe/simDES")
library(simDES)

# Check the documentation of the main function sim_DES
?sim_DES

# E.g. simulating 100 lineages over 25 million years with a rate shift 5.3 million years ago and sampling heterogeneity
Sim <- sim_DES(Time = 25, Step = 0.01, BinSize = 0.25, Nspecies = 100,
               SimD = c(0.2, 0.1, 0.3, 0.15),
               SimE = c(0.1, 0.1, 0.05, 0.15),
               SimQ = c(0.5, 0.4, 0.7, 0.8),
               Qtimes = 5.3, Ncat = Inf, alpha = 1)

# Write input table
write.table(Sim[[1]], ".../PyRate/example_files/DES_examples/DES_input_data/sim.txt",
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NaN")
```
<br>

## Running a DES analysis

PyRateDES requires the Python library *nlopt* for fitting models with Maximum likelihood. It can be installed with `pip install nlopt`.

### Basic DES analysis with constant rates

To launch PyRateDES open a Terminal window and browse to the PyRate directory 

`cd /path/to/PyRate`

The following command executes a DES analysis with dispersal, extinction and sampling rates that are constant through time but differ between both areas. The `-TdD` and `-TdE` commands specify time-dependent dispersal and extinction rates, respectively.

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE`

The output file will be named *Carnivora_1_0_TdD_TdE.log* and is saved where the input data was (here: .../example_files/DES_examples/Carnivora). The optional `-out`argument allows to add a user-defined name to the output.

The default settings specify Bayesian inference. We can (and in the cases of more complex models with time-variable rates we should) change the number of MCMC iterations and the sampling frequency. By default PyRateDES will run 100,000 iterations and sample and print the parameters every 100 iterations. Depending on the size of the data set you may have to increase the number iterations to reach convergence (in which case it might be a good idea to sample the chain less frequently to reduce the size of the output files). This is done using the commands `-n`, `-s`, and `-p`:

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -n 500000 -s 1000 -p 10000`

The same DES model can be fitted with Maximum likelihood by setting the algorithm to `-A 3` instead of using the default `-A 0`. This is typically faster than Bayesian inference but does not quantify the uncertainty of the model parameters.

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -A 3`

We can obtain the mean and 95% credible interval of all model parameters of a Bayesian analysis by summarizing its output:

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -sum .../example_files/DES_examples/Carnivora/Carnivora_1_0_TdD_TdE.log`

<br>

### DES analysis with heterogeneity in preservation rates across taxa

You can include differences in preservation rates across taxa. The command `-mG` specifies a model where the mean sampling rate across all taxa equals q and the heterogeneity is given by a discretized Gamma distribution with *n* categories. The default of four categories (`-ncat 4`) is usually sufficient to account for heterogeneity across lineages and a higher number increases computation time. Incorporating sampling heterogeneity improves the rate estimation. Sampling heterogeneity is computationally inexpensive to infer as it only adds a single free parameter to the model and should therefore be always included in DES models.

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -mG`

<br>

### Time variable model with rate shifts (Skyline model)

Dispersal, extinction, and preservation rates are allowed to shift at discrete moments in time, which are specified with the `-qtimes` argument. These shifts could be e.g. chronostratigraphic stages or series.

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -mG -qtimes 20.43 15.97 13.65 11.63 7.25 5.33 2.58 -n 1000001 -s 1000 -p 1000`

There are several optional constraints on dispersal, extinction, and preservation rates possible.

* `-symd`, `-syme`, and `-symq` constrain rates to be equal between areas. In a skyline model, rates are identical between areas but are allowed to differ among the time-strata defined with `-qtimes`

* `-constr` forces certain rates to be constant across time-strata while others are still allowed to vary over time. Indices define which rates should be constrained to be constant. `-constr 1` constrains the dispersal rate from area A to B, 2 dispersal B to A, 3 extinction in A, 4 extinction in B, 5 sampling in A, and 6 sampling in B. Several constraints can be combined e.g. `-constr 3 5 `.
<br>

### Covariate dependent dispersal and extinction models

PyRateDES2.py includes an upgraded version of the original DES model which allows more flexibility in time-variable dispersal and extinction models. You can use a time variable predictor (e.g. sea level or temperature) and model dispersal and/or extinction as a function of the predictor. The predictors should be tab-separated text files located in a seperate directory. See the file [sealevel.txt](https://github.com/dsilvestro/PyRate/blob/master//example_files/DES_examples/Carnivora/covariate_dispersal/sealevel.txt) in the example files. 

| age | sealevel |
| --- |:--------:|
0.000 | -19.636
0.002 | -23.272
0.004 | -20.727
0.006 | -51.636
0.010 | -114.181

You can use the same or different predictors for dispersal and extinction. For instance, you can test sea level as a predictor of dispersal and a climate proxy as a predictor for extinction. Several covariates could influence dispersal and/or extinction rates and should be located in the same directory. The arguments `-TdD`and `-TdE` should be omitted when covariate effects are inferred.

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -varD .../example_files/DES_examples/Carnivora/covariate_dispersal -varE .../example_files/DES_examples/Carnivora/covariate_extinction`

By default, all covariates are scaled to the range [0,1]. This behavior can be changed with the argument `-r`.

Covariate dependent models can be combined with `-qtimes` to allow sampling rates to vary over time and `-mG` to model heterogeneity in sampling acroos taxa. 

Moreover, several constraints on the covariate effect are possible:

* `-symCovD` and `-symCovE` constrain the covariate effect on dispersal and extinction rates to be symmetric for both areas via indices. E.g. `-symCovD 1` constrains the first covariate to have a symmetric effect on both dispersal rates while `-symCovD 2` applies to the second dispersal covariate.

* `-constrCovD_0` and `-constrCovE_0` set the covariate effect on dispersal or extinction to zero (i.e. no such effect of the covariate) via indices. E.g. `-constrCovD_0 1 4` removes through index 1 the effect of the first covariate on the dispersal rate from area A to B and through index 4 the covariate effect on dispersal from B to A.
<br>

### Diversity dependent dispersal and extinction models

Dispersal rate into an area could decline with the increase in diversity of the respective area and extinction rate may increase with the area's diversity. PyRateDES2.py allows to quantify and test these effects with the arguments `-DivdD`for diversity-dependent dispersal and `-DivdE`for diversity-dependent extinction.

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -DivdD -DivdE`

A different effect of diversity on extinction is that immigrating taxa drive resident taxa to extinction (e.g. invasion). This effect can be included in the DES model with the argument `-DdE`. Diversity dependent models can be combined with `-qtimes` and `-mG`.

Constraints are possible:

* `-symDivdD` and `-symDivdE` constrain the diversity effect on dispersal and extinction rates to be symmetric for both areas.

* `-constrDivdD_0` and `-constrDivdE_0` set the diversity effect on dispersal or extinction to zero through indices. E.g. `-constrDivdD_0 2` removes the effect for dispersal from area B to A and `-constrDivdE_0 1` specifies diversity-independent extinction in area A.
<br>

### Trait dependent dispersal and extinction models

In addition to covariate and/or diversity effects on dispersal and extinction rates, we can also infer the effect of continuous or categorical traits on these rates. Continuous traits should be a tab-separated text file. See the file [Body_mass_1.txt](https://github.com/dsilvestro/PyRate/blob/master//example_files/DES_examples/Carnivora/Body_mass_1.txt) in the example files. 

| scientificName | BodyMass |
| -------------- |:--------:|
Acinonyx | 93.604
Actiocyon | 3.435
Adcrocuta | 32.497
Adilophontes | 137.499
Aelurocyon | 1.731

The DES anaylsis can be launched with the following command:

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -traitD .../example_files/DES_examples/Carnivora/Body_mass_1.txt`

* `-traitD` and `-traitE` specifies the continuous traits influencing dispersal and extinction, respectively.

* `-logTraitD 0` or `-logTraitE 0` No trait transformation. By default, all traits are internally log-transformed.

Trait-dependent dispersal and extinction rates can be combined with an environmental influence on these rates and categorical traits (as well as different preservation models `-qtimes` and `-mG`). 

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -varE .../example_files/DES_examples/Carnivora/covariate_extinction -traitD .../example_files/DES_examples/Carnivora/Body_mass_1.txt`

Categorical traits (e.g. higher taxonomy like family) needs to be numerical coded. See the file [FamilyGenera.txt](https://github.com/dsilvestro/PyRate/blob/master//example_files/DES_examples/Carnivora/FamilyGenera.txt) in the example files. Additional column can be used to reflect the hierarchical taxonomy of the taxon.

| scientificName | Family |
| -------------- |:--------:|
Acinonyx | 6
Actiocyon | 1
Adcrocuta | 8
Adilophontes | 2
Aelurocyon | 10

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -catD .../example_files/DES_examples/Carnivora/FamilyGenera.txt`

* `-catD` and `-catE` specifies the categorical traits influencing dispersal and extinction, respectively.
<br>

## Summarizing and plotting the DES output

### Summarize model probabilities

The **mcmc.log** file can be used to obtain the mean and credible interval of the model parameters where the flag `-b` specifies the burnin (default 0; e.g. 1000 for the first 1000 MCMC generations).

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -sum .../example_files/DES_examples/Carnivora/Carnivora_1_0_TdD_TdE.log -b 1000`

### Plotting model output

With the help of the **marginal_rates.log** file we can plot the trajectory of the area-specific dispersal and extinction rates through time.

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -TdE -qtimes 5.33 -plot .../example_files/DES_examples/Carnivora/Carnivora_1_0_q_5.33_TdD_TdE_marginal_rates.log -b 1000 -plotCI 0.75 0.95`

This will generate an R script and a PDF file with the rates-through-time plots. This requires that R is installed on your PC to execute the shell command Rscript. If you are using Windows, please make sure that the path to Rscript.exe is included in the PATH environment variables (default in Mac/Linux).

* `-plotCI` allows to specify the width of the credible intervals to be plotted. Default is only one interval of 0.95 but several intervals can be plotted with different opacities.

The effect of environmental covariates, diversity, and traits can also be plotted:

`python ./PyRateDES.py -d .../example_files/DES_examples/Carnivora/Carnivora_1.txt -TdD -varE .../example_files/DES_examples/Carnivora/covariate_extinction -traitD .../example_files/DES_examples/Carnivora/Body_mass_1.txt -plot .../example_files/DES_examples/Carnivora/Carnivora_1_0_TdD_Eexp_TraitD_marginal_rates.log -b 5000`

