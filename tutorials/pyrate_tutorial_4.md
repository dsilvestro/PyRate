<img src="https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/PyRate_logo1024.png" align="left" width="80">  

# PyRate Tutorial \#4 MDB and ADE models
#### Daniele Silvestro – June 2018
***
#### Contents
* [Multivariate Birth-Death models](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#multivariate-birth-death-models-this-tutorial-is-work-in-progress)  
* [Age dependent extinction (ADE) model](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#age-depdendent-extinction-ade-model)
  
***

# Multivariate Birth-Death models (this tutorial is work in progress)

The MBD model allow the estimation of speciation and extinction rates as a function of multiple time-continuous variables [(Lehtonen, Silvestro et al. 2017)](https://www.nature.com/articles/s41598-017-05263-7). The model assumes linear or exponential functions linking the temporal variation of birth-death rates with changes in one or more variables.
Under the MBD model a correlation parameter is estimated for each variable (for speciation and extinction).

A Horseshoe prior algorithm (more details provided [here)](https://www.nature.com/articles/s41598-017-05263-7) is used to shrink around zero the correlation parameters, thus reducing the risk of over-parameterization and the need for explicit model testing. 

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

where `-m 1` specifies the type of correlation model, the options being `-m 0` for linear correlations and `-m 1` for exponential correlations. 

Other available options are:  
`-out outname` add a string to output file names   
`-rmDD 1` remove self-diversity dependence  
`-T 23` truncate at max time  
`plot <logfile>` plot marginal rates through time as predicted by the MBD model  

![Example RTT](https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/Ferns_MBD_short_run.png)


# Age dependent extinction (ADE) model

Testing for age dependent extinction is more complicated than fitting a Weibull distribution to the estimated longevities of species because that would not account for the unobserved species, which are likely the short lived ones ([Hagen et al. 2017](https://academic.oup.com/sysbio/article/doi/10.1093/sysbio/syx082/4563320/Estimating-Agedependent-Extinction-Contrasting)). The ADE model is implemented in its own function in PyRate and requires an [input file](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#generate-pyrate-input-file-option-1) containing all occurrences.
e.g.

`python PyRate.py <your_dataset> -ADE 1 -qShift epochs.txt`

where -ADE 1 specifies that you want to run the ADE model and -qShift is the  sets a preservation model with rate shifts ([TPP model](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#defining-the-preservation-model). Note that only homogeneous Poisson process (HPP) and time-variable Poisson process can be used with the ADE model. 

The ADE model assumes that extinction rates are only a function of species age and the mean rate does not change through time. Thus, ADE models should be tested within time windows with roughly constant speciation and extinction rates. In PyRate you can use the command `-filter` to drop all taxa outside a specified time range, e.g. 

`python PyRate.py <your_dataset> -ADE 1 -qShift epochs.txt -filter 23.03 5.3`

will only analyze taxa with all occurrences in the Miocene. The output file from an ADE analysis includes the estimated shape and scale parameters of the Weibull distribution. If the shape parameter is not significantly different from 1, then there is no evidence of age dependent extinction rates. Shape parameters smaller than 1 indicate that extinction rate is highest at the very beginning of a species life span and decreases with species age. Conversely, shape parameter values larger than one indicate that extinction rates increase with time since a species origination. 
