# PyRate Tutorial \#4
#### Daniele Silvestro – Jan 2018
***
Useful links:  
[PyRate code](https://github.com/dsilvestro/PyRate)  
[Paleobiology Database](https://paleobiodb.org)  
***

# Multivariate Birth-death models

The MBD model allow the estimation of speciation and extinction rates as a function of multiple time-continuous variables [(Lehtonen, Silvestro et al. 2017)](https://www.nature.com/articles/s41598-017-05263-7). The model assumes linear or exponential functions linking the temporal variation of birth-death rates with changes in one or more variables.
Under the MBD model a correlation parameter is estimated for each variable (for speciation and extinction).

A Horseshoe prior algorithm (more details provided [here)](https://www.nature.com/articles/s41598-017-05263-7) is used to shrink around zero the correlation parameters, thus reducing the risk of overparameterization and the need for explicit model testing. 

The MBD model is implemented in the program "PyRateMBD.py".

./PyRateMBD.py -d /example_files/Ferns_SE.txt -var /example_files/predictors_MBDmodel -m 1


`-m 0` Linear
`-m 1` Exponential
`-out outname`
`-var` directory with the variables
`-rmDD 1` remove self-diversity dependence
`-T 23` truncate at max time
`plot <logfile>`



Each oredictor should be provided as a tabxseparated table with header and two columns for time before present and predictor values: 

time | predictor
----- | -------
0	| 0.06
1	| 0.0665
2	| 0.073
3	| 0.0795
4	| 0.086
