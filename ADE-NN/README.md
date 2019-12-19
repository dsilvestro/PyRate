# Estimation of Age Dependent Extinction using Neural Networks (ADE-NN)

***

Codes and datasets used in the [Open Access paper](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.13441): Silvestro et al. A 450 million years long latitudinal gradient in age-dependent extinction. 

Future updates of the code and its documentation will be available on [GitHub](https://github.com/dsilvestro/PyRate).

***

## Requirements
Running the program **requires Python v.3** and the following libraries: 

numpy==1.16.3  
scipy==1.2.1   
keras=2.2.4   
sklearn==0.20.3   
tensorflow==1.13.1   
tensorflow==2.0.0  
matplotlib==3.0.3  


**NOTE**
The code is currently being updated to make it compatible with Tensorflow v.2, but for the time being it will only run using Tensorflow v.1.


## Data
The input data should be formatted as a text file with tab-separated columns (see Example_data). Tables contain four columns following the format of standard PyRate input files (see [example](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md#generate-pyrate-input-file-option-2)). **Note that the ADE-NN method as been only trained and tested on extinct species.** 


## Implementation
The ADE-NN model is implemented in the program `ADE-NN.py`. A pre-trained neural network is available as `pretrainedADENN` and includes 3 hidden layers. The empirical datasets analyzed here are available in the directory `occurrence_data`. To launch the ADE-NN analyses on the empirical data and reproduce the results shown in Table S4 you can use the following command:

```
python3 ADE-NN.py -layers 2 -loadNN pretrainedADENN -seed 1234 -data_dir full_path/lat_zones
```

where `-layers 2` specifies that 2 additional hidden layers should be used after the first default hidden layer and `-data_dir` specifies the directory with all input files. Using the seed `1234` ensures that the results match exactly those shown in Table S4 (but different seeds are not expected to significantly alter the results).

To run the analyses while accounting for potential taxonomic inflation you can use the following command:

```
python3 ADE-NN.py -layers 2 -loadNN pretrainedADENN -seed 1234 -data_dir full_path/lat_zones -tax_bias 0.2
```

where `-tax_bias 0.2` sets the taxonomic bias to 20%. These settings generate the results shown in Table S6. Setting `-tax_bias` to 0.1 and 0.3 will reproduce the results shown in Tables S5 and S7, respectively. 




