<img src="https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/PyRate_logo1024.png" align="left" width="80">

# PyRate Tutorial \#7

***  
#### Contents
* [BDNN model](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#the-bdnn-model)
* [Quick example](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#complete-example-for-the-impatient)
* [Setting up a BDNN dataset](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#setting-up-a-bdnn-dataset)
* [Rates through time plot](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#plotting-marginal-rates-through-time)
* [Effect plots](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#partial-dependence-plots)
* [Predictor importance](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#predictor-importance)
* [Combining replicates](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#combining-bdnn-files-across-replicates)
* [Return to Index](https://github.com/dsilvestro/PyRate/tree/master/tutorials#pyrate-tutorials---index)
***


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
## Complete example for the impatient
This is a short example of all the steps of the BDNN analysis. This will take you ca. 45 minutes and assumes some familiarity with PyRate. All steps, their optional arguments, and the output files are explained in their respective section below.

#### Move to your PyRate folder
```
cd /your/path/to/PyRate-master
```

Here, we have an [example dataset](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora) of Carnivora occurrences, their traits, paleotemperature, and a file specifying shifts in sampling rates through time. See the section [Setting up a BDNN dataset](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#setting-up-a-bdnn-dataset) for details.

#### Run a BDNN inference
```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -BDNNmodel 1 -trait_file ./example_files/BDNN_examples/Carnivora/Traits.txt -BDNNtimevar ./example_files/BDNN_examples/Carnivora/Paleotemperature.txt -mG -qShift ./example_files/BDNN_examples/Carnivora/Stages.txt -n 200001 -p 20000 -s 5000
```

#### Plot speciation and extinction rates through time
The `-plotBDNN` command will create a PDF file with the marginal rates through time (RTT).
```
python ./PyRate.py -plotBDNN ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -b 0.5
```

The optional argument `-b 0.5` discards 50% of the MCMC samples as burnin. Additional options to display the RTT for a subset of taxa are [detailed below](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#plotting-marginal-rates-through-time).

<img src="https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/BDNN/Carnivora_BDNN_RTT.png" alt="Rates through time" width="1000">
Rates through time plot for the Carnivora BDNN analysis obtained with the command `-plotBDNN`.


#### Display the influence of traits and paleotemperature on rates
We can create partial dependence plots (PDP) to visualize the influence of single predictors and all two-way interactions on speciation and extinction rates with the `-plotBDNN_effects` command.
```
python ./PyRate.py -plotBDNN_effects ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -BDNN_groups "{\"geography\": [\"Eurasia\", \"NAmerica\"], \"taxon\": [\"Amphicyonidae\", \"Canidae\", \"Felidae\", \"FeliformiaOther\", \"Hyaenidae\", \"Musteloidea\", \"Ursidae\", \"Viverridae\"]}" -b 0.5
```

The optional argument `-plotBDNN_transf_features` rescales z-transformed continuous traits and time-series predictors to their original scale. The `-BDNN_groups` is used to display categorical predictors with multiple unordered states, for instance, the family to which each taxon belongs in the same figure. See [Setting up a BDNN dataset](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#setting-up-a-bdnn-dataset) for details on trait encoding.

<img src="https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/BDNN/Carnivora_BDNN_family_speciation.png" alt="Family specific speciation rate" width="1000">
Carnivora families have different speciation rates according to the partial dependence plots.

&nbsp;

<img src="https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/BDNN/Carnivora_BDNN_temp_extinction.png" alt="Temperature dependent extinction" width="1000">
Lower temperatures are related to higher extinction rates according to the partial dependence plots. Ticks along the x-axis display the observed values of the predictor at the inferred extinction times of the species.



#### Obtain predictor importance
In the last step, we (a) assess if the variation in species-time-specific rates exceeds the expectation under a constant diversification process, and (b) rank the predictors according to their influence on speciation and extinction rates.
```
python ./PyRate.py -BDNN_pred_importance ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -BDNN_groups "{\"geography\": [\"Eurasia\", \"NAmerica\"], \"taxon\": [\"Amphicyonidae\", \"Canidae\", \"Felidae\", \"FeliformiaOther\", \"Hyaenidae\", \"Musteloidea\", \"Ursidae\", \"Viverridae\"]}" -b 0.5 -BDNN_nsim_expected_cv 10
```

We set the optional argument `-BDNN_nsim_expected_cv` to 10 (instead of the default 100) to safe some time for the impatient when getting the expected rate variation.

The `-BDNN_pred_importance` command creates seven files in the log files' folder. The `_coefficient_of_rate_variation.csv` file summarizes the variation in the inferred speciation and extinction rates among taxa and compares them with the upper 95% quantile of rate variation under a constant diversification process with the same root age and a similar number of taxa ± 25% as the analysed dataset. An output with higher empirical rate variation than expected permits to dig into which predictors are mainly causing this variation.

| rate | cv_empirical | cv_expected |
|:---- |:------------:|:-----------:|
speciation | 0.55 | 0.29
extinction | 0.82 | 0.34


The files `_sp_predictor_influence.csv` and `_ex_predictor_influence.csv` provide the ranked importance for the predictor variables (i.e. the features of the neural network) according to the consensus across three explainable artificial intelligence metrics. The main parts of the table are exemplified here for the three most important predictors of extinction:

| feature1 | feature1_state | posterior_probability | shap | delta_lik | rank |
|:-------- |:-------------- | ---------------------:| ----:| ---------:| ----:|
taxon | Amphicyonidae_Canidae | 0.9 | 0.17 | -72.6 | 2
taxon | Amphicyonidae_Felidae | 0.8 | 0.17 | -72.6 | 2
geography | Eurasia_NAmerica | 0.6 | 0.09 | -68.1 | 3
temp | none | 1.0 | 0.28 | -215.8 | 1

The column `feature1` lists the predictors, `feature1_state` shows the pairwise comparison for categorical predictors, `posterior_probability` quantifies the consistency of the direction of the predictors' effect (e.g. the proportion of the 10 sub-sampled MCMC generations genera of which the family Amphicyonidae had an higher extinction rate than Canidae), `shap` measures the effect size of the predictor (e.g. the extinction rate of Eurasian and North American carnivores differs by 0.09 units), `delta_lik` is the decrease in model likelihood when permuting the predictor, and finally the `rank` column provides the consensus among the individual importance ranking of these three metrics.

More details on this table, the remaining four output files when obtaining the predictor importance, and optional arguments for the `-BDNN_pred_importance` command are explained [below](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#predictor_importance).


---
## Setting up a BDNN dataset
The BDNN model requires occurrence data in the [standard PyRate format](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md). 
It additionally can use species and time-specific data. 
A tab-separated text file with species-specific trait data can be loaded in the analysis using the `-trait_table` command, while a table with time-series predictors can be loaded using the `-BDNNtimevar` command. 

We provide an [example dataset](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora) based on [Hauffe et al 2022 MEE](https://doi.org/10.1111/2041-210X.13845). This includes genus-level occurrence data of northern hemisphere Cenozoic carnivores, a table with a paleotemperature time series, and a table with lineage-specific traits: log-transformed body mass, taxonomic information (family-level classification), and continent of occurrence (Eurasia and North America). The tables are simple tab-separated text files with a header.

Note that to improve model convergence **continuous trait and time-series predictor should be z-transformed** (i.e. subtracting the mean and dividing by the standard deviation). The original mean and standard deviation can be stored in a tab-separated text file (see [`Backscale.txt`](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora/Backscale.txt) file for an example) and used in the plotting function to [display the results on the original scale](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#partial-dependence-plots).


#### Categorical traits
Categorical traits are allowed to be binary (i.e. two states of the trait), unordered, or have multiple states. 
A binary trait where a taxon can have only one of the two states (e.g. whether a taxon is aquatic) should be 0-1 encoded.

| scientificName | Aquatic |
| -------------- |:-------:|
Orcinus | 1
Cavia | 0


Unordered traits with more than two states should be one-hot encoded. For instance, the taxon's 'family assignment.

| scientificName | Felidae | Musteloidea | Ursidae |
| -------------- |:-------:|:-----------:|:-------:|
Acinonyx | 1 | 0 | 0
Actiocyon | 0 | 1 | 0
Agriotherium | 0 | 0 | 1


An unordered trait where a taxon can have more than one state should also be one-hot encoded. An example is the geographic distribution of taxa in two areas.

| scientificName | Eurasia | NAmerica |
| -------------- |:-------:|:--------:|
Acinonyx | 1 | 0
Actiocyon | 0 | 1
Agriotherium | 1 | 1


An ordered trait with more than two states should be encoded with integers. If there are many states, it would be best to center them in zero (i.e. subtracting the real-numbered value closest to the median of the states) and add the trait to the `Backscale.txt` for the `-plotBDNN_transf_features` argument with the value in the first row equal to the median and the value in the second row set to 1. For instance, for six taxa with the states 0, 0, 1, 2, 3, 4, we subtract 2 from each state because the median is 1.5.

| scientificName | Ordered_trait |
| -------------- |--------------:|
Taxon_0 | -2
Taxon_1 | -2
Taxon_2 | -1
Taxon_3 | 0
Taxon_4 | 1
Taxon_5 | 2

To have labels for the partial dependence plots beginning with zero, the file for the `-plotBDNN_transf_features` argument should then include a column like this:

| Ordered_trait |
|:-------------:|
| 2 |
| 1 |

#### Time-series of environmental variables
Time-series of e.g. environmental variables or biotic covariate for the `-BDNNtimevar` argument need to include **time** as the first column and the actual time-series variable in the following columns. These variables should be z-transformed to improve model convergence.

| Time | Temperature | Prey |
------:| -----------:| ----:|
0.00 | -1.16 | -1.34
0.09 | -2.71 | -0.52
0.14 | -3.39 | -0.47
0.33 | -1.08 | -0.01
1.17 | -1.44 | 0.23
2.91 | -0.77 | 0.68
7.35 | -0.22 | 1.19

The mean and standard deviation used to perform the z-transformation should be added to an optional tab-separated text file with the mean in the first row and the standard deviation in the second row. Adding temperature to the hypothetical use of the ordered categorical trait above, gives an `-plotBDNN_transf_features` file like this:

| Ordered_trait | Temperature |
|:-------------:| -----------:|
| 2 | 16.02 |
| 1 | 2.20 |


However, the `-plotBDNN_transf_features` is optional and does not change anything in the BDNN model inference. The effect plot using the `-plotBDNN_effects` command will be more meaningful though.

<img src="https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/BDNN/Carnivora_BDNN_backtransformation.png" alt="Backscaling z-transformation" width="1000">
Temperature effect on carnivore extinction rate with (right) and without (left) reversing the z-transformation of temperature with the `-plotBDNN_transf_features` argument when creating the effect plot from the same BDNN log file.

<br/><br/>

You can use different time-series for speciation and extinction by providing the paths to two text files using `-BDNNtimevar`. These files should have the same column names though. For instance, the emergance of humans could influence extinction rates but not on speciation rates.

```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -BDNNmodel 1 -trait_file ./example_files/BDNN_examples/Carnivora/Traits.txt -BDNNtimevar ./example_files/BDNN_examples/Carnivora/Advanced_examples/Paleotemperature_nohumans.txt ./example_files/BDNN_examples/Carnivora/Advanced_examples/Paleotemperature_humans.txt -fixShift ./example_files/BDNN_examples/Carnivora/Advanced_examples/Time_windows.txt -mG -qShift ./example_files/BDNN_examples/Carnivora/Stages.txt -out _different_timevar -n 200001 -p 20000 -s 5000
```


Time-series data can be provided in an already binned format. This needs to match the time bins of the neural network, which are by default 1 myr (but see [usage of custom bins](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#replacing-default-1-myr-bins)).

| Time | Temperature |
------:| -----------:|
0.0 | Mean value from 0.0-1.0 Ma
1.0 | Mean value from 1.0-2.0 Ma
2.0 | Mean value from 2.0-3.0 Ma


---
## Running a BDNN inference
To run a BDNN analysis we need to provide the occurrence file and use the command `-BDNNmodel 1`. By default the BDNN model will only use time as predictor, discretized in 1-myr bins. Additionally, traits should be subjected to the BDNN model using the `-trait_file` argument and time-series of e.g. environmental variables can be added with `-BDNNtimevar`. Time as a predictor can be omitted by including the setting `-BDNNtimetrait 0`, but this should be justified.

All standard commands to [define the preservation model](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#defining-the-preservation-model) can be used in combination with the BDNN model.
Here we will use the `-qShift` command to specify time bins for a time-variable preservation process and `-mG` to infer variation in sampling rates among species.

The analysis is launched as follows:
```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -BDNNmodel 1 -BDNNtimevar ./example_files/BDNN_examples/Carnivora/Paleotemperature.txt -trait_file ./example_files/BDNN_examples/Carnivora/Traits.txt -qShift ./example_files/BDNN_examples/Carnivora/Stages.txt -mG -n 20000001 -s 20000 -p 10000
```
where `-s` and `-n` are used to define sampling frequency and total of MCMC iterations, while `-p` sets the frequency of prints on your screen. Additional Hyper-parameters specifying the network structure and prior settings are listed under [Other options](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#other-options). 

As always in PyRate, the different replicates can be selected with the `-j` argument (e.g. `-j 2` for the second replicate). An analysis with the settings above will last approximately half a day. Output files of such a longer inference are included in the [Example_output](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora/Advanced_examples/Example_output).


#### Replacing default 1-myr bins
While origination and extinction times of the taxa are inferred in continuous time, the time effect of the BDNN and time-series are by default binned into 1-myr bins. If you want to use bin sizes you can use the `-fixShift` command to provide a file defining custom bins' boundaries. Custom bins can be useful to e.g. capture the higher temperature variability in the Pleistocene (instead of just taking the means from 0&mdash;1 and 1&mdash;2 with the default 1-myr bins). Moreover, larger bins could be useful for periods with few fossil occurrences and will result in faster BDNN runs. The `-fixShift` file for custom bins should have the same format as the one for the `-qShift` of the preservation model (i.e. no header, just ages). An example is provided in the [Advanced_examples](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora/Advanced_examples/Time_windows.txt)

|        |
|--------|
| 0.0117 |
| 0.126 |
| 0.781 |
| 1.8 |

Just include the `-fixShift` argument in the BDNN command:
```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -BDNNmodel 1 -BDNNtimevar ./example_files/BDNN_examples/Carnivora/Paleotemperature.txt -trait_file ./example_files/BDNN_examples/Carnivora/Traits.txt -fixShift ./example_files/BDNN_examples/Carnivora/Advanced_examples/Time_windows.txt -qShift ./example_files/BDNN_examples/Carnivora/Stages.txt -mG -out _time_windows -n 20000001 -s 20000 -p 10000
```

As with other PyRate analyse, the `-out` arguments allow us to add a suffix to the output files. We add here __time_windows_ to distinguish it from the previous BDNN runs. Output files of this BDNN model are included in the [Example_output](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora/Advanced_examples/Example_output) and can be compared with the other analyses.

---
## Output postprocessing

### Plotting marginal rates through time
The marginal speciation and extinction rates through time inferred by the BDNN model can be plotted using `-plotBDNN`:

```
python ./PyRate.py -plotBDNN ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -b 0.1
```

where `-b 0.1` specifies the burnin proportion. This uses the `_sp_rates.log` and `_ex_rates.log` files from the `pyrate_mcmc_logs` directory. The command will generate a PDF file and an R script with the rates-through-time plots, which will be saved in the `pyrate_mcmc_logs` directory. The R script file can be edited to customize the plot.

#### Marginal rates through time for a specific group of taxa
It is possible to plot the rates through time for a subset of the taxa, e.g. for a group of taxa sharing the same trait. The grouping needs to be provided in a tab-separated text file by assigning the taxon names to the group. For instance, the [example grouping](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora/RTT_groups.csv) specifies the subsets of carnivores that belonging to the families Amphicyonidae, Canidae or Musteloidea, occurring in Eurasia, weighing less than 10 kg, or just for the gluttonous eater (_Borophagus_).

```
python ./PyRate.py -plotBDNN ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -b 0.1 -plotBDNN_groups ./example_files/BDNN_examples/Carnivora/Advanced_examples/RTT_groups.csv
```

<img src="https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/BDNN/Carnivora_BDNN_RTT_Borophagus.png" alt="Rates through time for the genus Borophagus" width="1000">
Species-specific rates through time plot for the gluttonous eater (<i>Borophagus</i>).


---
### Partial dependence plots
We can now generate partial dependence plots to separate the individual effects of each predictor on the rates and the combined effects of each pair of predictors (to assess interactions). PDPs marginalize over the remaining predictors, i.e. cancelling out their effect and displaying only the effect attributed to the respective predictor(s). Although net diversification rate is not an inferred model parameter itself, we can display the effect on it by subtracting the extinction PDP from the speciation PDP. This is done using the `-plotBDNN_effects` command to load the `*mcmc.log` file. We additionally use the `-BDNN_groups` function to specify which variables are one-hot encoded and belong to the same class (e.g. all families belong to a class here named `taxon`).

```
python ./PyRate.py -plotBDNN_effects ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -BDNN_groups "{\"geography\": [\"Eurasia\", \"NAmerica\"], \"taxon\": [\"Amphicyonidae\", \"Canidae\", \"Felidae\", \"FeliformiaOther\", \"Hyaenidae\", \"Musteloidea\", \"Ursidae\", \"Viverridae\"]}" -b 0.1 -resample 100
```

We additionally specify the burnin fraction and the number of posterior samples considered in the PDP (`-resample`) because the log file may contain thousands of samples. However, 100 should be sufficient.

The command will generate a PDF file and an R script with the partial dependence plots, which will be saved in the `pyrate_mcmc_logs` directory. The R script file can be edited to customize the PDPs.

Note that the declaration of the `-BDNN_groups` includes *escaped quotes (i.e. \"). If you are using a Linux or Mac system, the expression can be simplified with single quotes for the outer quotes and double quotes for the trait and its states. 
```
-BDNN_groups '{"geography": ["Eurasia", "NAmerica"], "taxon": ["Amphicyonidae", "Canidae", "Felidae", "FeliformiaOther", "Hyaenidae", "Musteloidea", "Ursidae", "Viverridae"]}'
```

Optional arguments:

* `-b`: Burnin fraction (default 0.1).

* `-resample`: Number of post burnin MCMC samples (using all by default).

* `-plotBDNN_transf_features`: Tab-seperated text file with means and standard deviation to reverse z-transformation of continuous predictors, see [`Backscale.txt`](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora/Backscale.txt) file for an example.

* `-BDNN_groups`: String to plot all one-hot encoded states of a trait in the same figure.

* `-thread`: Number of CPU cores to be used for parallel calculation of importance metrics (only available on Linux or Mac systems); e.g. `-thread 2 0` to use two cores.

#### Partial dependence rates for more than two predictors

The partial dependence plots display the effect on speciation and extinction rates only for single predictors or the two-way interaction between two predictors. However, you can obtain a text file with the partial dependence rates for any set of predictors with the `-BDNN_interaction` command. This can last a while to compute and may use a lot of memory, especially for continuous predictors because for three continuous predictors we need to calculate 100 x 100 x 100 rates. The individuals predictors need to be specified with the `-BDNN_groups` argument.

This is an example to obtain partial dependence rates for the predictors body mass, temperature, and time (i.e. marginalizing over the effect of family and geographic distribution)
```
python ./PyRate.py -BDNN_interaction ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -BDNN_groups "{\"BodyMass\": [\"BodyMass\"], \"Temperature\": [\"Temperature\"], \"time\": [\"time\"]}" -b 0.5 -resample 3
```

It will create four text files in the *pyrate_mcmc_logs* folder (two for speciation and two for extinction): 
The *_pdp.csv* files gives in the first columns, which are named according to the `-BDNN_groups`, the combination of predictor values that were used to calculate the mean rate and its upper and lower 95% credible interval (in the last three columns).

| BodyMass | Temperature | time | mean | lwr | upr |
----------:| -----------:| ----:| ----:| ---:| ---:|
-0.73 | 10.73 | 0.5 | 0.16 | 0.16 | 0.11 | 0.25
-0.52 | 11.13 | 16.05 | 0.47 | 0.35 | 0.67

The other text file simply gives the predictor's values of the taxa at their inferred time of origination or extinction. This file can, for instance, be used to truncate the *_pdp.csv* file to the observed range of predictors.

To obtain partial dependence rates for more than two predictors, the `-BDNN_groups` can contain a mixture of continuous and categorical predictors (while this flag is usually reserved for one-hot encoded categorical traits).
```
python ./PyRate.py -BDNN_interaction ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -BDNN_groups "{\"BodyMass\": [\"BodyMass\"], \"Temperature\": [\"Temperature\"], \"taxon\": [\"Amphicyonidae\", \"Canidae\", \"Felidae\", \"FeliformiaOther\", \"Hyaenidae\", \"Musteloidea\", \"Ursidae\", \"Viverridae\"]}" -b 0.5 -resample 3
```

The `-BDNN_interaction` command can be also used to get species-specific speciation and extinction rates. Instead of calculating the rates for all combination of predictor values provided in the `-BDNN_groups`, we can get them for the taxa our occurrence dataset. This basically asks: *What would be the speciation (or extinction) rate of a taxon at the time of its origination (or extinction) if its family and its geographic distribution would not matter?* We only need to add the argument `-BDNN_interaction_fix`.
```
python ./PyRate.py -BDNN_interaction ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -BDNN_interaction_fix -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -plotBDNN_transf_features ./example_files/BDNN_examples/Carnivora/Backscale.txt -BDNN_groups "{\"BodyMass\": [\"BodyMass\"], \"Temperature\": [\"Temperature\"], \"time\": [\"time\"]}" -b 0.5 -resample 3
```

---
### Predictor importance

Finally, we can calculate the importance of each predictor using a combination of three metrics: 1) the marginal posterior probability of an effect, 2) the effect size of the predictor on the rates (SHAP values), and 3) the effect of the predictor on model fit (feature permutation).

This is done using the `-BDNN_pred_importance` command to load the `* mcmc.log` file. We use the `-BDNN_groups` function again to specify which variables belong in the same class.

```
python PyRate.py -BDNN_pred_importance ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -BDNN_groups "{\"geography\": [\"Eurasia\", \"NAmerica\"], \"taxon\": [\"Amphicyonidae\", \"Canidae\", \"Felidae\", \"FeliformiaOther\", \"Hyaenidae\", \"Musteloidea\", \"Ursidae\", \"Viverridae\"]}" -b 0.1 -resample 10 -BDNN_nsim_expected_cv 7 -BDNN_pred_importance_nperm 23
```

Optional arguments:

* `-b`: Burnin fraction (default 0.1).

* `-resample`: Number of post-burnin MCMC samples (using all by default).

* `-BDNN_nsim_expected_cv`: Number of constant diversification simulations to obtain the expected variation in species-time specific speciation and extinction rates (default: 100). The postprocessing step of getting the expected rate variation is ommitted when set to 0.

* `-BDNN_pred_importance_nperm`: Sets the number of feature permutations (default: 100).

* `-BDNN_pred_importance_interaction`: Add this argument to additionally rank two-way interaction of predictors. Depending on the number of predictors, this will be much slower.

* `-BDNN_pred_importance_window_size`: The feature permutations automatically corrects for time bins of different sizes (see [usage of custom bins](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#replacing-default-1-myr-bins)). However, this takes very long if the shortest bin is very small (e.g. the Holocene). The automatic correction can be adjusted to something slightly larger e.g. `-BDNN_pred_importance_window_size 0.1`.

* `-BDNN_groups`: String to calculate combined importance for a group of features e.g. for one-hot encoded states of a categorical trait. Works, however, also for groups of continuous traits (but not in combination with `-BDNN_pred_importance_interaction`)

* `-BDNN_mean_shap_per_group`: Use the mean instead of the sum across all states of a `-BDNN_groups` to get the overall importance of the group. Could be more robust when a trait has many states.

* `-thread`: Number of CPU cores to be used for parallel calculation of importance metrics (only available on Linux or Mac systems); e.g. `-thread 2 0` to use two cores.


The `-BDNN_pred_importance` command will generate two tab-separated tables with the estimated importance and ranking of each predictor on speciation and extinction rates. It will also generate a PDF file and an R script with the lineage-specific speciation and extinction rates and an estimation of how they are affected by the predictors' values. 


---
### Other options

A set of **Hyper-parameters** can be used to define the architecture of the neural network implemented in the BDNN.

`-BDNNnodes 16 8`: Defines the number of layers and nodes in each layer (default: two layers of 18 and 8 nodes); 30 20 10 would specify three layers with 30, 20, and 10 nodes.

`-BDNNoutputfun`: Activation function of the output layer: 0) abs, 1) softPlus, 2) exp, 3) relu 4) sigmoid 5) sigmoid_rate (default=5)

`-BDNNactfun`: Activation function of the hidden layer(s): 0) tanh, 1) relu, 2) leaky_relu, 3) swish, 4) sigmoid (default=0)

`-BDNNreg`: Specifies whether the regularization layer should be omitted by setting it to -1.0. The default value of 1.0 for the truncated exponential prior assigns the highest probability of no effect on the BDNN predictors. Higher positive values increase this weight. By default the same t~reg~ is used for regularizing speciation and extinction rates. This can be made independent by providing two values `-BDNNreg 1.0 1.0`. `-BDNNreg 1.0 -1.0` would turn on regularization for speciation but not extinction.


---
### Specifying custom predictors

A regular BDNN analysis with traits specified by `-trait_file` and paleoenvironmental variables included with the `-BDNNtimevar` command assumes that species traits do not change over time and that all species experience the same environmental conditions over time. It is possible to relax this assumption by subjecting custom build tables to the analysis. For instance, temperature trajectories could differ among geographic areas or humans could influence extinction while an effect on speciation is not plausible.
Such an analysis can be set-up using custom tables, which have a similar format as the standard `-trait_file` (i.e., species in rows and traits, including paleoenvironment, in columns, but no column with taxon names). For an analysis that includes the default effect of time, one custom table per time-bin is needed. When no boundaries between time-bins are specified with `-fixShift` (see [example](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_bdnn.md#replacing-default-1-myr-bins)), this means one table per 1 million years. Otherwise, the number of tables need to equal the number of boundaries. The file name of the custom tables need to reflect the time-bins, for instance, *01.txt* for the most recent one, *02.txt* for the following (older) bin etc.
Custom tables are subjected to the BDNN analysis using the `-BDNNpath_taxon_time_tables` command, which takes the path to a folder containing the custom tables. If a single path is provided, the custom tables are used as predictors for speciation and extinction rates. If two paths are given, the first one is for the predictors of speciation rates and the second for extinction rates.
No additional `-trait_file` and `-BDNNtimevar` should be invoked.

The following example uses custom tables with humans being present during the past 126,000 years in Eurasia, while humans were present in North America mainly from the Holocene on. This spatial-temporal overlap with humans could influence the extinction rate but not speciation. Additionally, trajectories of paleotemperature are continent-specific.
```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -BDNNmodel 1 -BDNNpath_taxon_time_tables ./example_files/BDNN_examples/Carnivora/Advanced_examples/load_predictors/speciation ./example_files/BDNN_examples/Carnivora/Advanced_examples/load_predictors/extinction -fixShift ./example_files/BDNN_examples/Carnivora/Advanced_examples/Time_windows.txt -qShift ./example_files/BDNN_examples/Carnivora/Stages.txt -mG -out _taxon_time_tables -n 20000001 -s 20000 -p 10000
```

We use the `-out` argument again to add a suffix to the output files. Output files of this BDNN model are called __taxon_time_tables_ and are included in the [Example_output](https://github.com/dsilvestro/PyRate/tree/master/example_files/BDNN_examples/Carnivora/Advanced_examples/Example_output).

<img src="https://github.com/dsilvestro/PyRate/blob/master/example_files/plots/BDNN/Carnivora_BDNN_humans.png" alt="Partial dependence plot of human effect" width="1000">
Partial dependence plot of the influence of the human spatial-temporal overlap with Carnivora on their extinction rate. The effect on extinction rate is higher in North America, where this effect was only allowed in the custom BDNN predictors for taxa living until the Holocene. The effect of humans on Eurasian carnivores was permitted during the past 126,000 years. <i>Humans1.0</i> indicates spatial-temporal overlap.

&nbsp;

To help setting-up the correct number of custom tables and get their format right, PyRate allows to export the tables containing traits and environmental predictors from a BDNN analysis with the `-BDNNexport_taxon_time_tables` flag. These tables could then be modified using a text editor or spreadsheet software.
```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -BDNNmodel 1 -BDNNtimevar ./example_files/BDNN_examples/Carnivora/Paleotemperature.txt -trait_file ./example_files/BDNN_examples/Carnivora/Traits.txt -fixShift ./example_files/BDNN_examples/Carnivora/Advanced_examples/Time_windows.txt -BDNNexport_taxon_time_tables
```

---
### Restoring BDNN analysis

BDNN inferrences may need more MCMC generations than expected to reach the stationarity distribution of the Markov chain and the selected number of generation `-n` might be to small. In this case, you can restore the MCMC chain from the last sample using the `-restore_mcmc` argument.

```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -restore_mcmc ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs/Carnivora_occs_1_G_BDS_BDNN_16_8TVc_mcmc.log -BDNNmodel 1 -trait_file ./example_files/BDNN_examples/Carnivora/Traits.txt -BDNNtimevar ./example_files/BDNN_examples/Carnivora/Paleotemperature.txt -mG -qShift ./example_files/BDNN_examples/Carnivora/Stages.txt -n 200001 -p 20000 -s 5000
```

---
### Combining BDNN files across replicates

To account for age uncertainty in fossil occurrences, you should create multiple replicates with randomly sampled ages (see [PyRate tutorial 1](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md)). The BDNN model can be inferred for these replicates independently and their output files can be combined to obtain e.g. a single rate through time plot for all replicates or obtain the predictor importance across replicates.

As always in PyRate, the different replicates can be selected with the `-j` argument. For instance, the Carnivora example dataset contains three replicates and the 2nd can be run with the following line:

```
python ./PyRate.py ./example_files/BDNN_examples/Carnivora/Carnivora_occs.py -j 2 -BDNNmodel 1 -trait_file ./example_files/BDNN_examples/Carnivora/Traits.txt -BDNNtimevar ./example_files/BDNN_examples/Carnivora/Paleotemperature.txt -mG -qShift ./example_files/BDNN_examples/Carnivora/Stages.txt -n 200001 -p 20000 -s 5000
```

To combine log files from different replicates into one you can use the command:

```
python ./PyRate.py -combBDNN ./example_files/BDNN_examples/Carnivora/pyrate_mcmc_logs -b 20
```

where `path_to_your_log_files` specifies the directory where the log files are (e.g., pyrate_mcmc_logs). 
This command generates five output files named “combined_[n]_mcmc.log”, "combined_[n]_per_species_rates.log", "combined_[n]_sp_rates.log", "combined_[n]_ex_rates.log", and combined_[n].pkl, where [n] is the number of combined replicates.
The flag `-b 0.25` specifies that the first 25% of all MCMC samples from each file should be excluded as burnin – the appropriate number of burnin samples to be excluded should be determined after inspecting the mcmc.log files, e.g. using Tracer. To avoid producing too large combined files you can sub-sample each log file, using the flag `-resample`. If there are different BDNN analyses e.g. with different sets of predictors in the same pyrate_mcmc_logs folder, the respective analyses can be selected with the `-tag` argument. For instance, `-tag taxon_time_tables` instructs PyRate to combine all analyses whose file name contains taxon_time_tables.

