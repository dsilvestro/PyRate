<img src="https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/PyRate_logo1024.png" align="left" width="80">  

# The BD-NN models (work in progress)

#### June 2021
---

The BD-NN models use a neural network to model changes in origination and extinction rates as a function of one or more discrete and/or continuous traits and as a function of time. 
The BD-NN model is available using a constant baseline rate (`-A 0`) or with time variable rate within fixed time frames (`-A 0 -fixShift <file>`).
There are currently three flavors of the model:

1. Trait dependent time-homogeneous model
`PyRate.py <data> -trait_file <file> -A 0 -BDNNmodel 1`

2. Trait dependent time-variable model 
`PyRate.py <data> -trait_file <file> -A 0 -BDNNmodel 1 -fixShift <file>`
This model estimates trait-dependent effects while accounting for time-variable rates

3. Trait dependent time-variable model with time as a trait
`PyRate.py <data> -trait_file <file> -A 0 -BDNNmodel 1 -fixShift <file> -BDNNtimetrait <value>`
This model estimates trait-dependent effects while accounting for time-variable rates and additionally includes time as a trait. The idea is that this way traits can have aa different effect on rates at different times. If the `<value>` is set to a positive number it will be used to rescale time as a trait such that `rescaled_time = time * value`. The time used as a trait is the mid age within each time bin defined through `-fixShift`. 


4. Trait dependent time-variable model with time as a trait and constant baseline rates
`PyRate.py <data> -trait_file <file> -A 0 -BDNNmodel 1 -fixShift <file> -BDNNtimetrait <value> -BDNNconstbaseline 1`
This model is similar to model \#3 but the baseline rates are constant and set to 1. Thus the rates and all the rate variation is captured by the neural network. The good thing of this model is that the number of parameters is independent of the number of time bins, potentially allowing for a high resolution in how time is affecting the rates. 
This in practice can be achieved by setting via `-fixShift` many small time slices, e.g. 1 per Myr. Note that a high number of time bins will increase computing time. 
The potential limitation is that all rate variation is captured through the NN, which might mean that a higher number of nodes is necessary to make a good job (but this has not been verified yet). 

#### Additional settings
There are currently two available activation functions for the output layer (while for the hidden layer it is set to ReLU), and which ensure that the resulting rates are positive: the softPlus function (`-BDNNoutputfun 0`) and the exponential function (`-BDNNoutputfun 1`). While the choice of activation function might have an effect on efficiency of the MCMC and convergence, this has not yet been explored and might depend on the dataset. 

`-BDNNnodes <int> (default=3)`: number of nodes in the hidden layer





