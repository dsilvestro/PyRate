# PyRateDES v.2 [work in progress]

#### Setup data
After loading `pyrate_DES_utilities.R` library,
follow the template in `pyrate_DES_input_example_script.R` to generate input files.


#### To launch PyRateDES open a Terminal window and browse to the PyRate directory 

`cd /path/to/pyrate/code`

#### Time variable model (rate shifts)
Launch a maximum likelihood analysis with shifts in preservation, dispersal, and extinction rates at 5.3 and 2.6 Ma.

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -TdD -TdE`

The `-TdD` and `-TdE` commands specify time-dependent dispersal and extinction rates, respectively.

The command `-A 2` specifies that you want to use a maximum likelihood algorithm. 

Note that if `input_data.txt` is in not in he same directory as `PyRateDES2.py` you need to specify its full path. The output files will be saved where the 


#### Covariate D/E models
You can use a time variable predictor (e.g. sea level or temperature) and model dispersal and/or extinction as an exponential function of the predictor. 

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -varD predictor_file.txt -varE predictor_file.txt`

Note that if `predictor_file.txt ` is in not in he same directory as `PyRateDES2.py` you need to specify its full path. The format of the predictor file is a tab-separated text file with column headers as shown in PyRate's repository (`/PhanerozoicTempSmooth.txt`).

#### Mixed D/E models
You can use different predictors for dispersal and extinction, e.g.

`./PyRateDES2.py -d input_data.txt -A 2 -qtimes 5.3 2.6 -varD predictor_file1.txt -varE predictor_file2.txt`

For instance you can test sea level as a predictor of dispersal and a climate proxy as a predictor for extinction. 

##### The likelihoods of different models (with different predictors or rate shifts) can be compared to perform model testing, for example using AIC scores. 















