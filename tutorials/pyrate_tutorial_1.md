# PyRate Tutorial \#1
#### Daniele Silvestro – Nov 2017
***
Useful links:  
[PyRate code](https://github.com/dsilvestro/PyRate)  
[PyRate wiki](https://github.com/dsilvestro/PyRate/wiki)  
[Paleobiology Database](https://paleobiodb.org)  
[Tracer](http://tree.bio.ed.ac.uk/software/tracer/)
***

## Generate PyRate input file (option 1)
1. **Download fossil occurrences** for a clade from the Paleobiology Database. E.g. search for the genus *Canis* and save it as a cvs file, e.g. using the file name *Canis_pbdb_data.csv*. Before downloading the file, check the box "Show accepted names only" in the "Select by taxonomy" section, and uncheck the box "Include fmetadata at the beginning of the output" in the "Choose output options" section.

2. **Launch R** by opening a Terminal window and typing `R` then hit Enter. On Mac or Windows you can use the R GUI app or RStudio. 

3. **Load the *pyrate_utilities.r* file** e.g. by typing:  
  `source(".../PyRate-master/pyrate_utilities.r")`. Note that the full path of a file is here indicated as `.../` for simplicity. In most operating systems you can drag and drop the *pyrate_utilities.r* file onto the R console to paste the full path. 

4. **Check which species are extant today** (if any), as this information must be provided when running a diversification rate analysis. All species unless otherwise specified will be considered as extinct. Define a vector of extant species by typing in R:   
  `extant_dogs = c("Canis rufus","Canis lupus","Canis aureus","Canis latrans","Canis mesomelas","Canis anthus","Pseudalopex gymnocercus","Canis adustus","Canis familiaris")`

5. **Parse the raw data and generate PyRate input file**. Here we are going to use an automated function to extract fossil occurrence data from PBDB raw table and save it in a PyRate-compatible input file. Type in R:   
  `extract.ages.pbdb(file= ".../Canis_pbdb_data.csv",extant_species=extant_dogs)`  
  **IMPORTANT NOTE:** this function does not check for synonyms, typos, etc. You can specify two additional options.  The option `replicates` (by default set to 1) resamples the ages of each fossil occurrence from the respective temporal range. If you use `replicates=10`, ten replicated datasets will be saved in a single PyRate input file. These replicates can then be analyzed (individually) and the combined results will account for age uncertainties in the fossil record. The option `cutoff` can be used to remove occurrences with an age range greater than a given temporal range. For instance using `cutoff=10` will remove all occurrences in which the difference between max age and min age is larwger than 10 Myr. 

6. **Check PyRate input files.** The previous step generated two files: a text file (\*\_SpeciesList.txt) containing the list of all species included in the dataset and a python file (\*\_PyRate.py) with all occurrences formatted for a PyRate analysis. We can check the python file directly in PyRate to get a few summary statistics:  
  a. Open a Terminal window
  b. Browse to the Pyrate folder, e.g.:  
    `cd ".../PyRate-master"`
  c. Launch PyRate with the following arguments:  
  `python PyRate.py '.../Canis_pbdb_data_PyRate.py' -data_info`  
***

## Generate PyRate input file (option 2)
1. **Prepare fossil occurrence table.** You can prepare a table with fossil occurrence data in a text editor or a spreadsheet editor. The table must include 4 columns including 1) Taxon name, 2) Status specifying whether the taxon is "extinct" or "extant", 3) Minimum age, and 4) Maximum age. The table should have a header (first row) and **one row for each fossil occurrence**. Min and max ages represent the age ranges of each fossil occurrence, typically based on stratigraphic boundaries. One additional column can be included in the table indicating a trait value (e.g. estimated body mass) that can be used PyRate analyses.  
2. **Launch R** as explained above. 
3. **Load the *pyrate_utilities.r* file** as explained above.  
4. **Parse the raw data and generate PyRate input file**. Type in R:   
  `extract.ages(file="…/PyRate/example_files/Ursidae.txt", replicates=10)` This function includes `replicates` and `cutoff` options as the `extract.ages.pbdb()` function described above (option 1). 




***
## Estimation of speciation/extinction rates through time

#### Defining the preservation model
By default, PyRate assumes a **non-homogeneous Poisson process of preservation** (NHPP) in which preservation rates change during the lifespan of each lineage following a bell-shaped distribution (Silvestro et al. 2014 Syst Biol). Alternatively, a **homogeneous Poisson process (HPP)**, in which the preservation rate is constant through time, is also available, using the command `-mHPP`:

`python PyRate.py .../Canis_pbdb_data_PyRate.py -mHPP`

Both NHPP and HPP models can be again paired with a **Gamma model of rate heterogeneity**, which enables us to account for heterogeneity in the preservation rate across lineages. This option only adds a single parameter to the model and should be used for all empirical data sets. To set the Gamma model we add the flag `-mG`:

`python PyRate.py .../Canis_pbdb_data_PyRate.py -mG` [NHPP model]  
`python PyRate.py .../Canis_pbdb_data_PyRate.py -mHPP -mG` [HPP model]


**Time-variable Poisson process (TPP)**. An alternative model of preservation assumes that preservation rates are constant within a predefined time frame, but can vary across time frames (e.g. geological epochs). This model is particularly useful if we expect rate heterogeneity to occur mostly through time, rather than among lineages.

We can set up a model in which preservation rates are estimated independently within each geological epoch by using the command `-qShift` and providing a file containing the times that delimit the different epochs (an example is provided in `PyRate-master/example_files/epochs_q.txt`):  

`python PyRate.py .../Canis_pbdb_data_PyRate.py -qShift .../epochs_q.txt`  

Finally, a **TPP + Gamma model** can be used to **incorporate both temporal and across-lineages variation in the preservation rates**. This is perhaps the most realistic preservation model currently available in PyRate and is set with the following commands:

`python PyRate.py .../Canis_pbdb_data_PyRate.py -qShift .../epochs_q.txt -mG`  




#### Analysis setup
Here we describe the main settings for a standard analysis of fossil occurrence data using PyRate. The analysis will estimate:

1. **origination and extinction times** of each lineage  
2. **preservation rate** and its level of **heterogeneity**  
3. **speciation and extinction rates** through time. 

Temporal rate variation is introduced by rate shifts. The number and temporal placement of shifts are estimated from the data using the BDMCMC algorithm.

The analysis requires a **PyRate input file** generated by the R function described above. The first argument we need to provide is the input file:

`python PyRate.py .../Canis_pbdb_data_PyRate.py`

In most operating systems (including Mac OS, Windows, and Ubuntu) you can drag and drop the input file onto the terminal window to paste the full path and file name. 

_PyRate includes default settings for all parameters except for the input file. While several parameters should be changed only when experiencing convergence issues, there are a few that are very important as they change the basic model assumptions or the length of the analyses - see below_.

Since the input file generated in the previous steps included 10 randomized replicates of the fossil ages, we can specify which replicate we want to analyze. **Ideally, we should analyze multiple randomized replicates and combine the results to incorporate dating uncertainties** in our rate estimates. To specify the which replicate we want to analyze, we use the flag `-j` followed by the replicate number. For instance using:

`python PyRate.py .../Canis_pbdb_data_PyRate.py -mHPP -mG -j 1`

we set the analysis to consider the first replicate.

The BDMCMC algorithm is the default setting in PyRate and we do not need to specify any additional parameters to estimate speciation and extinction rates through time. We can (and in some cases should) however change the number of BDMCMC iterations and the sampling frequency. By default PyRate will run 10,000,000 iterations and sample the parameters every 1,000 iterations. Depending on the size of the data set you may have to **increase the number iterations to reach convergence** (in which case it might be a good idea to sample the chain less frequently to reduce the size of the output files). This is done using the commands `-n` and `-s`:

`python PyRate.py .../Canis_pbdb_data_PyRate.py -mG -n 20000000 -s 5000`

Under these settings PyRate will run for 20 million iterations sampling every 5,000. Thus the resulting log files (see below) will include 4,000 posterior samples.  

**Another algorithm available uses Reversible Jump MCMC (RJMCMC) to look for rate shifts. Preliminary simulations show that RJMCMC outperforms BDMCMC (paper in preparation) in accuracy and power: see Tutorial \# for more details.** 

#### Output files
The PyRate analysis described above produces three output files, saved in a folder named *pyrate_mcmc_logs* in the same directory as the input file:

######  sum.txt  
   Text file providing the complete list of settings used in the analysis.  
######  mcmc.log   
Tab-separated table with the MCMC samples of the posterior, prior, likelihoods of the preservation process and of the birth-death (indicated by *PP_lik* and *BD_lik*, respectively), the preservation rate (*q_rate*), the shape parameter of its gamma-distributed heterogeneity (*alpha*), the number of sampled rate shifts (*k_birth*, *k_death*), the time of origin of the oldest lineage (*root_age*), the total branch length (*tot_length*), and the times of speciation and extinction of all taxa in the data set (*\*_TS* and *\*_TE*, respectively). When using the TPP model of preservation, the preservation rates between shifts are indicated as *q_0, q_1, ... q_n* (from older to younger).
###### marginal_rates.log 
Tab-separated table with the posterior samples of the marginal rates of speciation, extinction, and net diversification, calculated within 1 time unit (typically Myr). 


#### Summarize the results
The log files can be opened in the program **Tracer** to check if the MCMC has converged and determine the proportion of burnin. 

The **mcmc.log** file can be used to calculate the sampling frequencies of birth-death models with different number of rate shifts. This is done by using the PyRate command `-mProb` followed by the log file:

`python PyRate.py -mProb .../Canis_pbdb_data_mcmc.log -b 200`

where the flag `-b 200` indicates that the first 200 samples will be removed (i.e. the first 200,000 iterations, if the sampling frequency was set to 1,000). This command will provide a table with the relative probabilities of birth-death models with different number of rate shifts. 


The **marginal_rates.log** file can be used to generate rates-through-time plots using the function `-plot`:

`python PyRate.py -plot .../Canis_pbdb_data_marginal_rates.log -b 200`

This will generate an R script and a PDF file with the RTT plots showing speciation, extinction, and net diversification through time. A slightly different flavor of the RTT plot can be obtained using the flag `-plot2` instead of `-plot`. 



#### Combine log files across replicates
To combine log files from different replicates into one you can use the command:

`PyRate.py -combLog path_to_your_log_files -tag mcmc -b 100`

where `path_to_your_log_files` specifies the directory where the log files are; `-tag mcmc` sets PyRate to combine all files that contain _mcmc.log_ in the file name (you can use different tags if you need); and `-b 100` specifies that the first 100 samples from each log file should be excluded as burnin – the appropriate number of burnin samples to be excluded should be determined after inspecting the log files, e.g. using Tracer.

To generate a rates-through-time plot that combines all replicates, you can use the command:

`PyRate.py -plot path_to_your_log_files -tag marginal_rates -b 100`

This will combine all the _marginal_rates.log_ files in the specified directory and combine the results in a single plot.​ Different tags can be used to determine which files are to be combined.    

***
## Speciation/extinction rates within fixed time bins
#### Analysis setup
PyRate can also fit birth-death models in which the **number and temporal placement of rate shifts is fixed a priori**, e.g. based on geological epochs. In this case a file with the predefined times of rate shifts must be provided using the command `-fixShift`. The format of this file is very simple and an example is available here: `.../PyRate-master/example_files/epochs.txt`. This model assumes half-Cauchy prior distributions for speciation and extinction rates between shifts, with a hyper-prior on the respective scale parameter to reduce the risk of over parameterization. 
To enforce fixed times of rate shifts we use the following command:

`python PyRate.py .../Canis_pbdb_data_PyRate.py -fixShift .../epochs.txt`

The other options described above to set preservation model, length of the MCMC, and sampling frequency are also available in this case.

#### Summarize the results
Running PyRate with fixed times of rate shifts produces the same 3 output files described in the previous analysis. The main difference is in the *mcmc.log* file where we will no longer have the estimate number of rate shifts (*k_birth*, *k_death*) as these are fixed. However, the log file now includes speciation/ extinction rates between shifts (named *lambda_0, lambda_1,* ... and  *mu_0, mu_1, ...*, respectively), and the estimated scale parameters of the half-Cauchy prior distributions assigned to speciation and extinction rates, indicated by *hypL* and *hypM*, respectively.

RTT plots can be generated as in the previous analysis using the command `-plot` (or `-plot2`) followed by the path to the *marginal_rates.log* file and setting the number of samples be discarded as burnin (e.g. `-b 100`).



## Setting fixed shifts at the boundaries, while searching for rate shifts between them

Sometimes fossil data sets are truncated by max and min boundaries (for instance because occurrences are only available within a time window). This can cause apparent rate shifts at the edges of the time range, which reflect the sampling bias. In this case, you can fix _a priori_ times of rate shift based on the known temporal boundaries of the data set and estimate the rates within the time window, ignoring what happens beyond the boundaries.  This feature can be combined with the RJMCMC algorithm (`-A 4`) to infer rate shifts within the allowed time window:

`python PyRate.py <data_set> -A 4 -edgeShift 18 2`

where -A 4 sets the RJMCMC algorithm (that looks for rate shifts), and `-edgeShift 18 2` sets fixed times of shifts at times 18 and 2. With these settings the algorithm will only search for shifts within this time range.
You can also set only a max age boundary shift using:

`python PyRate.py <data_set> -A 4 -edgeShift 18 0`

or a min age boundary shift using:

`python PyRate.py <data_set> -A 4 -edgeShift inf 0`

When summarizing the results, only rates estimated within the time window should be considered.