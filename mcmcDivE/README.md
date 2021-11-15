# Bayesian estimation of diversity trajectories

Estimation of corrected diversity trajectories using `mcmcDivE` requires a PyRate occurrence file (`-d`), a text file with the time bins used to infer preservation rates (`-q`), and a log file with posterior samples of the preservation rates inferred from a previous PyRate analysis (`-m`):

`python3 mcmcDivE.py -d path_to_input_file/pyrate_occ_data.py -q path_to_input_file/intervals.txt -m path_to_input_file/pyrate_mcmc.log`

**NOTE** that like PyRate v.3, mcmcDivE requires Python v.3. 

### Additional (optional) commands are:  
`-n`: the number of MCMC iterations
`-s`: sampling frequency
`-p`: print frequency
`-b`: number of equal-size time bins for the diversity trajectory
`-j`: number of randomized ages of fossil occurrences in the PyRate input files (these will be resampled across MCMC samples)
`-N`: number of extant taxa

### Example code
```
python3 mcmcDivE.py -d example/Rhinos.py -q example/epochs_q.txt -m example/pyrate_mcmc_logs/Rhinos_Grj_mcmc.log -b 50 -j 1 -N 5
```
### Plot estimated diversity trajectories:

Plotting functions to summarize the estimated diversity trajectories are implemented in the R script `plot_mcmcDivE_results.R `:  

```
source("path_to_script/plot_mcmcDivE_results.R")
log_file = "path_to_log_file/output_mcmcDivE.log"
plot_diversity(log_file)
```

























