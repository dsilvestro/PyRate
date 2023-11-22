<img src="https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/PyRate_logo1024.png" align="left" width="80">  

# The BDNN model 

The birth-death neural-network model modulates speciation and extinction rates per lineage and through time as a function of 

1. **time**
2. one or multiple **categorical and/or continuous traits** (e.g. diet, body mass, geographic range)
3. one or more **time-dependent variables** (e.g. paleotemperature)
4. **phylogenetic relatedness** (e.g. classification into higher taxa or phylogenetic eigenvectors)

As the function is based on a fully connected feed-forward neural network, it is not based on *a priori* assumptions about its shape. For instance, it can account for non-linear and non-monotonic responses of the rates to variation in the predictors.  
It can also account for any interactions among the predictors. 

The parameters of the BDNN model are estimated jointly with the origination and extinction times of all taxa and the preservation rates. 
The output can be used to estimate rate variation through time, across species, and to identify the most important predictors of such variation and their individual or combined effects. 

---

### Setting up a BDNN dataset
The BDNN model requires occurrence data in the [standard PyRate format](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md). 
It additionally can use species and time specific data. 
A table with species-specific trait data can be loaded in the analysis using the `-trait_table` command, while a table with time-series predictors can be loaded using the `-BDNNtimevar` command. 

We provide an [example dataset](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora) based on [Hauffe et al 2022 MEE](https://doi.org/10.1111/2041-210X.13845). This includes genus level occurrence data of northern hemisphere Cenozoic carnivores, a table with a paleotemperature time series, and a table with lineage-specific traits: log-transformed body mass, taxonomic information (family-level classification), and continent of occurrence (Eurasia and North America). The tables are simple tab-separated text files with a header. 

Note that the log Body mass and paleotemperature data are z-transformed (and the original mean and standard deviation are stored in the `Backscale.txt` file for plotting). 


---
### Running a BDNN analysis
To run a BDNN analysis we need to provide the occurrence file and use the command `-BDNNmodel 1`. By default the model will only use time as predictor, discretizing time in 1-myr bins. If you want to use different time bins (e.g. of different sizes) you can use the `-fixShift` command to provide a file defining the bins' boundaries.

All [standard commands to define the preservation model](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#defining-the-preservation-model) can be used in combination with the BDNN model.
Here we will use the `-qShift` command to specify time bins for a time-variable preservation process. 

The analysis is launched as follow:

```
python PyRate.py .../Carnivora_occs.py -fixShift .../Time_windows.txt -BDNNmodel 1 -BDNNtimevar .../Paleotemperature.txt -qShift .../Stages.txt -mG -A 0 -trait_file .../Traits.txt -s 10 -n 1000

```
where `-s` and `-n` are used to define sampling frequency and MCMC iterations. **Note** that `...` should be replaced with the absolute path to the files. 



## Output postprocessing

### Plotting the marginal rates through time
The marginal speciation and extinction rates through time inferred by the BDNN model can be plotted using the standard `-plotRJ` [command](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_3.md#plot-rates-through-time-and-rate-shifts):

```
python PyRate.py -plotRJ .../pyrate_mcmc_logs -b 0.1
```

where `-b 0.1` specifies the burnin proportion. 
The command will generate a PDF file and an R script with the rates-through-time plots, which will be saved in the `pyrate_mcmc_logs` directory. The R script file can be edited to customize the plot. 


### Partial dependence plots
We can now generate partial dependence plots to separate the individual effects of each predictor on the rates and the combined effects of each pair of predictors (to assess interactions). This is done using the `-plotBDNN_effects` command to load the `*mcmc.log` file and the `-plotBDNN_transf_features` to load the `Backscale.txt` file (to rescale correctly the traits in the plots). We additionally use the `-BDNN_groups` function to specify which variables belong in the same class (e.g. all families belong to a class here named `taxon`. 

```
python PyRate.py -plotBDNN_effects .../pyrate_mcmc_logs/Carnivora_1_G_BDS_BDNN_16_8TVc_mcmc.log -plotBDNN_transf_features .../Backscale.txt -BDNN_groups "{\"geography\": [\"Eurasia\", \"NAmerica\"], \"taxon\": [\"Amphicyonidae\", \"Canidae\", \"Felidae\", \"FeliformiaOther\", \"Hyaenidae\", \"Musteloidea\", \"Ursidae\", \"Viverridae\"]}" -b 0.1 -resample 100
```

We additionally specify the burnin fraction and the number of posterior samples considered in the PDP (`-resample`). 

The command will generate a PDF file and an R script with the partial dependence plots, which will be saved in the `pyrate_mcmc_logs` directory. 
The R script file can be edited to customize the PDPs. 
 

### Predictor importance

Finally we can calculate the importance of each predictor using a combination of three metrics: 1) the marginal posterior probability of an effect, 2) the effect size of the predictor on the rates (SHAP values), and 3) the effect of the predictor on model fit (feature permutation).

This is done using the `BDNN_pred_importance` command to load the `* mcmc.log` file. The number of permutations and posterior samples can be adjusted using the flags `-BDNN_pred_importance_nperm` and `-resample`, respectively. We use again the `-BDNN_groups` function to specify which variables belong in the same class and to speed up the analysis we can use `-BDNN_pred_importance_only_main` to limit the importance estimation to single predictors (i.e. without testing for combinations of multiple predictors). 


```
python PyRate.py -BDNN_pred_importance.../pyrate_mcmc_logs/Carnivora_1_G_BDS_BDNN_16_8TVc_mcmc.log -BDNN_groups "{\"geography\": [\"Eurasia\", \"NAmerica\"], \"taxon\": [\"Amphicyonidae\", \"Canidae\", \"Felidae\", \"FeliformiaOther\", \"Hyaenidae\", \"Musteloidea\", \"Ursidae\", \"Viverridae\"]}" -b 0.1 -resample 1 -BDNN_pred_importance_nperm 10 -BDNN_pred_importance_only_main
```

This command will generate two tab-separated tables with the estimated importance and ranking of each predictor on speciation and extinction rates. It will also generate a PDF file and an R script with the lineage-specific speciation and extinction rates and an estimation of how they are affected by the predictors' values. 


### Other options

A set of **Hyper-parameters** can be used to define the architecture of the neural network implemented in the BDNN. 


`-BDNNnodes 16 8`: defines the number of layers and nodes in each layer (default: two layers of 18 and 8 nodes)

`-BDNNoutputfun`: Activation function of the output layer: 0) abs, 1) softPlus, 2) exp, 3) relu 4) sigmoid 5) sigmoid_rate (default=5)

`-BDNNactfun`: Activation function of the hidden layer(s): 0) tanh, 1) relu, 2) leaky_relu, 3) swish, 4) sigmoid (default=0)


### Specifying custom predictors

A regular BDNN analysis with traits specified by `-trait_file` and paleoenvironmental variables included with the `-BDNNtimevar` command assumes that species traits do not change over time and all species experience the same environmental conditions over time. It is possible to relax this assumption by subjecting custom build tables to the analysis. For instance, temperature trajectories could differ among geographic range or humans could influence extinction while an effect on speciation is not plausible.
Such an analysis can be set-up using custom tables, which have the same format as the standard `-trait_file` (i.e., species in rows and traits, including paleoenvironment, in columns). For an analysis that includes an effect of time (i.e., invoked by the default setting `-BDNNtimetrait -1`), one custom table per time-bin is needed. When no boundaries between time-bins are specified with `-fixShift`, this means one table per 1 million years. Otherwise, the number of tables need to equal the number of boundaries. The file name of the custom tables need to reflect the time-bins, for instance, *01.txt* for the most recent one, *02.txt* for the following (older) bin etc.
Custom tables are subjected to the BDNN analysis using the `-BDNNpath_taxon_time_tables` command, which takes the path to a folder containing the custom tables. If a single path is provided, the custom tables are used as predictors for speciation and extinction rates. If two paths are given, the first one is for the predictors of speciation rates and the second for extinction rates.
No `-trait_file` and `-BDNNtimevar` should be provided.

The following example uses custom tables with humans being present during the past 500,000 years in Eurasia but not in North America, which could influence the extinction rate but not speciation. Additionally, trajectories of paleotemperature are continent specific.
```
python PyRate.py .../Carnivora_occs.py -fixShift .../Time_windows.txt -BDNNmodel 1 -BDNNpath_taxon_time_tables .../load_predictors/speciation .../load_predictors/extinction -qShift .../Stages.txt -mG -A 0  -s 10 -n 1000
```

