<img src="https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/PyRate_logo1024.png" align="left" width="80">  

# PyRate Tutorial \#4
#### Jan 2020
***
#### Contents
* [Multivariate Birth-Death models](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#multivariate-birth-death-models-this-tutorial-is-work-in-progress)  
* [Trait-dependent diversification models (Covar model)](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#trait-correlated-diversification)
* [Age dependent extinction (ADE) models](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#age-dependent-extinction-ade-models)

* [Return to Index](https://github.com/dsilvestro/PyRate/tree/master/tutorials#pyrate-tutorials---index) 
***

# The multiple-clade diversity-dependent model (MCDD)
The original MCDD model described [here](https://www.pnas.org/content/112/28/8684) is implemented in the program `PyRateMCDD.py` and requires a single input file (see also [example file](https://github.com/dsilvestro/PyRate/blob/master/example_files/Carnivores_MCDD.txt)), which is a tab-separated table with 4 columns (or more if you have multiple replicates) and one row per species:

* The first column identifies the clade the species belongs to, starting from 0, 1, 2, etc.  
* The second column is a species identifier  
* The third and fourth columns are the times of origination and extinction of the species as inferred in a previous PyRate analysis (these can be generated using the `-ginput` command as shown [here](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_2.md#generate-input-file-for-pyratecontinuous))

If you have multiple replicates of the initial PyRate analysis, these can be added to the MCDD input file (see example file).

One the input file is ready, you can run an analysis using:

`PyRateMCDD.py -d /example_files/Carnivores_MCDD.txt -n 10000000 -s 10000`

where the flags `-n` and `-s` specify the number of MCMC iterations and the sampling frequency, respectively. By default, the program analyzes all clades at once. However, you can also analyze a single clade at time (i.e. the birth-death process of a specific clade with diversity dependencies from its own diversity and the diversity of all other clades) using the flag `-c`. For example 

`PyRateMCDD.py -d /example_files/Carnivores_MCDD.txt -n 10000000 -s 10000 -c 0` 

will estimates the diversity dependent effects from all clades affecting the speciation and extinction of the first clade. 

the MCDD model implements a Bayesian variable selection algorithm that can switch on and off the diversity dependent parameters depending on how important they are to explain rate variation using indicators that can take a value of 0 or 1. An alternative method that can be used to infer multiple-clade diversity dependence is implemented in the multivariate birth death model (MBD) described below and uses the horseshoe prior algorithm to perform variable selection. The performance of the two methods was compared in [this paper](http://www.evolutionary-ecology.com/abstracts/v18/3010.html), suggesting that the latter might be preferable.  

# The Multivariate Birth-Death model (MBD)

The MBD model allow the estimation of speciation and extinction rates as a function of multiple time-continuous variables [(Lehtonen, Silvestro et al. 2017)](https://www.nature.com/articles/s41598-017-05263-7). The model assumes linear or exponential functions linking the temporal variation of birth-death rates with changes in one or more variables.
Under the MBD model a correlation parameter is estimated for each variable (for speciation and extinction).

A **Horseshoe prior algorithm** (more details provided [here)](https://www.nature.com/articles/s41598-017-05263-7) is used to shrink around zero the correlation parameters, thus reducing the risk of over-parameterization and the need for explicit model testing. 

**Alternatively, gamma hyper-priors** can be used to constrain the correlation parameters and prevent over-parameterization (see [below](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#additional-options)). This option should be preferred when testing only few variables (e.g. 2-4 correlates).

The MBD model is implemented in the program `PyRateMBD.py` and requires as main input file a [table with estimated speciation and extinction times](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_2.md#generate-input-file-for-pyratecontinuous). It additionally requires a set of predictors provided as separate files in a single directory.
Each predictor should be provided as a tab-separated table with a header and two columns for time before present and predictor values, e.g.

time | predictor
----- | -------
0	| 0.06
1	| 0.0665
2	| 0.073
3	| 0.0795
4	| 0.086

Example files are available [here](https://github.com/dsilvestro/PyRate/tree/master/example_files/predictors_MBDmodel).

To launch an MBD analysis, you must provide the input file and the path to all predictor files:

`./PyRateMBD.py -d /example_files/Ferns_SE.txt -var /example_files/predictors_MBDmodel -m 1`

where `-m 1` specifies the type of correlation model, the options being `-m 0` for exponential correlations (default) and `-m 1` for linear correlations.
The flag `-var` is used to specify the path to a folder containing all predictors.

### Using the MBD model to run multiple-clade diversity dependent analysis
To use the MBD model in a multiple-clade diversity dependence analysis you should provide the diversity trajectories of all clades as predictors. Diversity trajectories files in the correct format can be generated using PyRate's `-ltt` [command](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_2.md#plot-range-through-diversity-trajectory-lineage-through-time-plot).

### Additional options  
* `-out outname` add a string to output file names   
* `-rmDD 1` remove self-diversity dependence (by default included in the analysis) 
*  `-minT 2.58` truncate at min time (e.g. at 2.58 Ma)
* `-maxT 23` truncate at max time (e.g. at 23 Ma)
* `-hsp 0` use Gamma hyper-priors instead of the Horseshoe prior
* `-n 10000000`  MCMC iterations
* `-s 5000`      sampling frequency

The MBD analysis uses the Horseshoe prior by default. However, this can be replaced with gamma hyper-priors on the precision parameters (1/variance) of the Gaussian priors on the correlation parameters. This is done adding the flag: `-hsp 0`.

### Summarizing the results
The command `-plot <logfile>` can be used to generate plots of the marginal speciation and extinction rates through time  as predicted by the MBD model. When plotting the results of the MBD analyses, the input data, the directory containing all predictors, and the correlation model (linear or exponential) must be specified:

`./PyRateMBD.py -d /example_files/Ferns_SE.txt -var /example_files/predictors_MBDmodel -m 1 -plot Ferns_SE_0_lin_MBD.log`

When using the Horseshoe prior, the `plot` function also computes the shrinkage weights (_Wl_ and _Wm_ for speciation and extinction, respectively) for all predictors. The shrinkage weights quantify the statistical support for each correlation factor. The correlation parameters are indicated by _Gl_ and _Gm_ in the log files and their posterior estimate is also indicated in the plots produced by the `-plot` command (they can also easily be obtained by opening the _mcmc.log_ file in [Tracer](http://tree.bio.ed.ac.uk/software/tracer/)).

![Example RTT](https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/Ferns_MBD_short_run.png)



# Trait correlated diversification 
PyRate implements birth-death models in which speciation and extinction rates change in a lineage-specific fashion as a function of an estimated correlation with a continuous trait (Covar models). The model is described [here](http://onlinelibrary.wiley.com/doi/10.1111/2041-210X.12263/abstract). 


### Providing a trait file
With the command `-trait_file` you can provide a table (tab-separated text file) with trait values for the species in your fossil data set. The first column of the table should include the species names (identical to those used in the PyRate file), the second column provides the trait values ([see example file](https://github.com/dsilvestro/PyRate/blob/master/example_files/Ursidae_bm.txt)). Species for which trait data are not available can be omitted from the table. Alternatively, they can be included in the table with trait value `NA`. These species will be still included in the analysis, but their trait value will be imputed by the MCMC.

Trait values can (often should) be log-transformed. This can be deon using the command `logT`:  
`-logT 0` trait is not transformed  
`-logT 1` trait is log_e transformed  
`-logT 2` trait is log10 transformed


### Setting the Covar model
Use the command `-mCov` to set Covar models in which the birth-death rates (and preservation rate) vary across lineages as the result of a correlation with a continuous trait, provided as an observed variable, based on estimated correlation parameters (cov_sp, cov_ex, cov_q).
Examples:  
`-mCov 1` correlated speciation rate  
`-mCov 2` correlated extinction rate  
`-mCov 3` correlated speciation and extinction rates  
`-mCov 4` correlated preservation rate  
`-mCov 5` correlated speciation, extinction, preservation rates  

The default prior on the correlation parameters is a normal distribution centered in 0 and with standard deviation = 1. The standard deviation can be modified using the command `-pC`, e.g. `-pC 0.1` sets the standard deviation to 0.1. Alternatively, the standard deviation of the normal prior on the correlation parameter can be estimated directly from the data (using hyper-priors; more details are provided [here](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2016.2361)). This is done setting `-pC 0`. **Note** that this option is generally preferred when more than one correlation parameters are estimated (i.e. with `-mCov 3` or `-mCov 5`).

A typical analysis is launched with the following command:

`./PyRate.py Ursidae_example_PyRate.py -trait_file Ursidae_bm.txt -mCov 5 -logT 10 -pC 0`

Example files available [here](https://github.com/dsilvestro/PyRate/blob/master/example_files).




# Age dependent extinction (ADE) models

### Bayesian implementation of the ADE model
NOTE: An alternative implementation of an Age Dependent Extinction model using deep neural networks (named ADE-NN) is described in this [Open Access paper](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.13441) and the code implementing it is available [here](https://github.com/dsilvestro/PyRate/tree/master/ADE-NN).


Testing for age dependent extinction is more complicated than fitting a Weibull distribution to the estimated longevities of species because that would not account for the unobserved species, which are likely the short lived ones ([Hagen et al. 2017](https://academic.oup.com/sysbio/article/doi/10.1093/sysbio/syx082/4563320/Estimating-Agedependent-Extinction-Contrasting)). 
The ADE model is implemented in its own function in PyRate and requires an [input file](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#generate-pyrate-input-file-option-1) containing all occurrences.
e.g.

`python PyRate.py <your_dataset> -ADE 1 -qShift epochs.txt`

where -ADE 1 specifies that you want to run the ADE model and -qShift is the  sets a preservation model with rate shifts ([TPP model](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#defining-the-preservation-model). Note that only homogeneous Poisson process (HPP) and time-variable Poisson process can be used with the ADE model. 

The ADE model assumes that extinction rates are only a function of species age and the mean rate does not change through time. Thus, ADE models should be tested within time windows with roughly constant speciation and extinction rates. In PyRate you can use the command `-filter` to drop all taxa outside a specified time range, e.g. 

`python PyRate.py <your_dataset> -ADE 1 -qShift epochs.txt -filter 23.03 5.3`

will only analyze taxa with all occurrences in the Miocene. The output file from an ADE analysis includes the estimated shape and scale parameters of the Weibull distribution. If the shape parameter is not significantly different from 1, then there is no evidence of age dependent extinction rates. Shape parameters smaller than 1 indicate that extinction rate is highest at the very beginning of a species life span and decreases with species age. Conversely, shape parameter values larger than one indicate that extinction rates increase with time since a species origination. 


# The Birth-Death Chronospecies (BDC) model

Fossil and phylogenetic data can be jointly analyzed under the BDC model as described by [Silvestro, Warnock, Gavryushkina & Stadler 2018](https://www.nature.com/articles/s41467-018-07622-y). This analysis requires two input files: a standard PyRate input dataset (that can be generated as explained [here](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#generate-pyrate-input-file-option-2); see also [examples](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDC_model)).

To run a joint analysis of fossil and phylogenetic data you should provide the standard PyRate input file (occurrence data) and a tree file (NEXUS format):

`python PyRate.py example_files/BDC_model/Feliformia.py -tree example_files/BDC_model/Feliformia.tre`

Note that this function requires the [Dendropy library](https://dendropy.org). 

By default the analysis assumes a constant rate birth-death model with independent rate parameters between fossil and phylogenetic data. 
The flag `-bdc` enforces **compatible speciation and extinction rates under the BDC model**, whereas the flag `-eqr` sets the **rates to be equal**.

To run under the **BDC skyline model** you can use the `-fixShift` command as explained [here](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#speciation-and-extinction-rates-within-fixed-time-bins), for example:

`python PyRate.py example_files/BDC_model/Feliformia.py -tree example_files/BDC_model/Feliformia.tre -fixShift example_files/epochs.txt -bdc`

This command sets up an analysis with rate shifts at the epochs boundaries under the BDC compatible model.

Plotting functions in R to plot the results of the BDC and BDC-skyline models are available [here](https://github.com/dsilvestro/PyRate/blob/master/plot_functions_BDC_model.R).


