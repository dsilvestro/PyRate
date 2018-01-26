# PyRate Tutorial \#4
#### Daniele Silvestro – Jan 2018
***
Useful links:  
[PyRate code](https://github.com/dsilvestro/PyRate)  
[Paleobiology Database](https://paleobiodb.org)  
***

# Multivariate Birth-death models

The MBD model allow the estimation of speciation and extinction rates as a function of multiple time-continuous variables [(Lehtonen, Silvestro et al. 2017)](https://www.nature.com/articles/s41598-017-05263-7). The model assumes linear or exponential functions linking the temporal variation of birth-death rates with changes in one or more variables.

The MBD model is implemented in the program "PyRateMBD.py".

./PyRateMBD.py -d /Users/danielesilvestro/Software/PyRate_github/example_files/Ferns_SE.txt -var /Users/danielesilvestro/Software/PyRate_github/example_files/predictors_MBDmodel -m 1


`-m 0` Linear
`-m 1` Exponential
`-out outname`
`-var` directory with the variables
`-rmDD 1` remove self-diversity dependence
`-T 23` truncate at max time
`plot <logfile>`
