# PyRate Tutorial \#3 RJMCMC
#### Feb 2018
***
#### Contents
* [Setting up an analysis using RJMCMC](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_3.md#setting-up-an-analysis-using-rjmcmc)  
* [RJMCMC output](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_3.md#rjmcmc-output)  
* [Summarize model probabilities](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_3.md#summarize-model-probabilities)    
* [Plot rates through time and rate shifts](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_3.md#rates-through-time-and-rate-shifts)  
* [Combine log files from multiple replicates](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_3.md#combine-log-files-from-multiple-replicates)  

Useful links:  
[PyRate code](https://github.com/dsilvestro/PyRate)  [PyRate wiki](https://github.com/dsilvestro/PyRate/wiki)
***

# Estimate rate variation using Reversible Jump MCMC

## Setting up an analysis using RJMCMC
We have recently implemented a new algorithm in Pyrate that uses RJMCMC (Green 1995) instead of BDMCMC (described in Silvestro et al. 2014 and available through the command `-A 2`). Although the paper describing the algorithm is still in preparation, the method is already available and simulations show that it is more accurate than BDMCMC.

An analysis with RJMCMC is set up using the command `-A 4`. Note that this algorithm is available both for occurrence data
e.g.:

`python PyRate.py .../Canis_pbdb_data_PyRate.py -A 4 -mHPP -mG`

and for data sets with fixed times of origination and extinction (see also tutorial \#2):

`python PyRate.py -d Canidae_1_G_se_est.txt -A 4`

You can find more details about preparing input files [here]
(https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#generate-pyrate-input-file-option-1)
and [here](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#generate-pyrate-input-file-option-2)

## RJMCMC Output
RJMCMC analyses by default produce 4 output files:
####  1) sum.txt  
   Text file providing the complete list of settings used in the analysis.  
#### 2)  mcmc.log   
Tab-separated table with the MCMC samples of the posterior, prior, likelihoods of the preservation process and of the birth-death (indicated by *PP_lik* and *BD_lik*, respectively), the preservation rate (*q_rate*), the shape parameter of its gamma-distributed heterogeneity (*alpha*), the number of sampled rate shifts (*k_birth*, *k_death*), the time of origin of the oldest lineage (*root_age*), the total branch length (*tot_length*), and the times of speciation and extinction of all taxa in the data set (*\*_TS* and *\*_TE*, respectively, if estimated). When using the TPP model of preservation, the preservation rates between shifts are indicated as *q_0, q_1, ... q_n* (from older to younger).
#### 3) sp\_rates.log, ex\_rates.log
Tab-separated text files providing sampled rates and times of rate shifts. Note that, because the number of shifts is likely to change throughout the RJMCMC, the number of columns in the text file is not fixed. This file can be analyzed using the command `-plotRJ` to obtain plots of the marginal rates through time and the most probable temporal placement of rate shifts, if any (see below). 

#### 4) marginal\_rates.log (old format)
An alternative output file ([the **marginal_rates.log** file described in tutorial \#1](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#output-files))  can be obtained instead, using the command `-log_marginal_rates 1` when setting up the analysis. Note, that this output can be processed using the command `-plot` to obtain rates-through-time plots (see Tutorial #1), but it cannot be analyzed using the command `plotRJ`, to assess the temporal placement of significant rate shifts.

## Summarize model probabilities
The **mcmc.log** file can be used to calculate the sampling frequencies of birth-death models with different number of rate shifts. This is done by using the PyRate command `-mProb` followed by the log file:

`python PyRate.py -mProb .../Canis_pbdb_data_mcmc.log -b 200`

where the flag `-b 200` indicates that the first 200 samples will be removed (i.e. the first 200,000 iterations, if the sampling frequency was set to 1,000). This command will provide a table (printed on screen) with the relative probabilities of birth-death models with different number of rate shifts. 


## Plot rates through time and rate shifts
The **sp_rates.log** and **ex_rates.log** files can be used to generate rates-through-time plots using the function `-plotRJ`:

`python PyRate.py -plotRJ .../path_to_log_files/ -b 200`

This will generate an R script and a PDF file with the RTT plots showing speciation and extinction rates through time. It will also show histograms with the inferred times of rate shifts and calculate Bayes Factors to help determining the time when a rate shift is supported by significant posterior probability. The histograms include two horizontal dashed lines showing the thresholds for positive evidence of a rate shift (bottom line: logBF = 2) and for strong evidence of a rate shift (top line: logBF = 6). Thus, any point in the histogram showing sampling frequencies for a rate shift exceeding the thresholds indicate a time of significant rate change.

Note that **this command requires the path to the log files (i.e. not the files themselves)**. If multiple compatible files are found in the directory, they will be plotted individually. All individual plots are however combined in a single R script and PDF file. To limit the plotting function to one or few selected log files, the command `-tag` can be used. For instance using the command `-tag Canis_pbdb` will result in RTT plots being computed only for log files containing "*Canis_pbdb*" in the file name.

Additional commands are avaiable to adjust settings of the `plotRJ` function. The command `-grid_plot` defined the size of the temporal bins utilized to calculate marginal rates and times of rate shift. The default is `-grid_plot 1`, but this can be changed to any arbitrarily small value.

The command `-root_plot` can be used to truncate the plot to a given maximum age. For instance using `-root_plot 10` will only show marginal rates in the most recent 10 time units. Finally, the command `-logT 1` can be added to plot log10-transformed rates-through-time.

![Example RTT](https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/RTT_plot_RJMCMC.png)


## Combine log files from multiple replicates
PyRate includes a utility function to combine output files from different runs into one file. Assuming that all output files form the previous analyses are in the same pyrate mcmc logs directory, the log files are combined using: 

`python PyRate.py -combLog .../pyrate_mcmc_logs -b 1000 -tag mcmc` 
`python PyRate.py -combLog .../pyrate_mcmc_logs -b 1000 -tag sp_rates`
`python PyRate.py -combLog .../pyrate_mcmc_logs -b 1000 -tag ex_rates`

 where: `-combLog .../pyrate_mcmc_logs` provides the full path to the log files, `-b 1000` specifies that the first 1,000 samples should be removed as burn-in, `-tag x` specifies that all files containing x in the file name should be combined. These commands generate output files named “combined\_[n]\_mcmc.log”, "combined\_[n]\_sp\_rates.log", and "combined\_[n]\_ex\_rates.log", where [n] is the number of combined replicates. 
 
For the `mcmc.log` file you can select which columns you want to include in the combined log file. For instance, adding the flag `-col_tag posterior root_age death_age` will combine only the three columns with headers `posterior`,`root_age`, and `death_age` (while ignoring all other parameters). Note that only `root_age`, and `death_age` are needed to run the `-plotRJ` function.

To avoid producing too large combined files you can sub-sample the log file, using the flag `-resample`. For example the command

`python PyRate.py -combLog .../pyrate_mcmc_logs -b 1000 -tag mcmc -resample 100` 

specifies that 100 random samples should be taken from each replicate and saved into the combined log files. 

