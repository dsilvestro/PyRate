<img src="https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/PyRate_logo1024.png" align="left" width="80">  

### PyRate is a program to estimate speciation, extinction, and preservation rates from fossil occurrence data using a Bayesian framework.

[![DOI](https://zenodo.org/badge/21620870.svg)](https://zenodo.org/badge/latestdoi/21620870)

---
**The latest version of PyRate uses Python v.3.** 
To upgrade Python visit: [https://www.python.org/downloads/](https://www.python.org/downloads/). Older versions of PyRate for Python v.2 are available [here](https://github.com/dsilvestro/PyRate/tree/master/PyRate_for_Python2).

---

PyRate is licensed under a [AGPLv3 License](https://tldrlegal.com/license/gnu-affero-general-public-license-v3-(agpl-3.0)#summary).

#### The program's documentation is available here: 
* [PyRate Tutorials](https://github.com/dsilvestro/PyRate/tree/master/tutorials) (Wiki pages are no longer being updated)
* [System requirements](https://github.com/dsilvestro/PyRate/wiki#compatibility-and-installation) to run the program and [Instructions](https://github.com/dsilvestro/PyRate/blob/master/pyrate_lib/fastPyRateC/README.md) to install the fastPyRateC library 
* For questions, suggestions or bugs [contact us](mailto:pyrate.help@gmail.com)


---


#### The main methods are described here:

* Silvestro D., Salamin N., Antonelli A., Meyer X. (2019) Improved estimation of macroevolutionary rates from fossil data using a Bayesian framework. Paleobiology,
[doi:10.1017/pab.2019.23](https://doi.org/10.1017/pab.2019.23)

* Silvestro D., Schnitzler J., Liow L.H., Antonelli A., Salamin N. (2014) Bayesian Estimation of Speciation and Extinction from Incomplete Fossil Occurrence Data. Systematic Biology, [63, 349-367](https://academic.oup.com/sysbio/article/63/3/349/1650079).

* Silvestro D., Salamin N., Schnitzler J. (2014) PyRate: A new program to estimate speciation and extinction rates from incomplete fossil record. Methods in Ecology and Evolution, [5, 1126-1131](http://onlinelibrary.wiley.com/doi/10.1111/2041-210X.12263/abstract).
 
* See tutorials [1](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_1.md), [2](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_2.md), and [3](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_3.md)


#### The multivariate birth-death models (MCDD and MBD) are described here: 

* Silvestro D., Antonelli A., Salamin N., Quental T. B. (2015) The role of clade competition in the diversification of North American canids. PNAS, [112, 8684-8689](http://www.pnas.org/content/112/28/8684).

* Silvestro D., Pires M. M., Quental T., Salamin N. (2017) Bayesian estimation of multiple clade competition from fossil data. Evolutionary Ecology Research, 	[18:41-59](http://evolutionary-ecology.com/abstracts/v18/3010.html).

* Lehtonen S., Silvestro D., Karger D. N., Scotese C., Tuomisto H., Kessler M., Peña C., Wahlberg N., Antonelli A. (2017) Environmentally driven extinction and opportunistic origination explain fern diversification patterns. Scientific Reports, [7:4831](https://www.nature.com/articles/s41598-017-05263-7).

* See tutorial [4](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md)


#### The dispersal-extinction-sampling model (DES) is described here:

* Silvestro D., Zizka A., Bacon C. D., Cascales-Minana B., Salamin N., Antonelli, A. (2016) Fossil Biogeography: A new model to infer dispersal, extinction and sampling from paleontological data. Philosophical Transactions of the Royal Society B [371:20150225](http://rstb.royalsocietypublishing.org/content/371/1691/20150225).

* Hauffe T., Pires M.M., Quental T.B., Wilke T., Silvestro, D. (2022) A quantitative framework to infer the effect of traits, diversity and environment on dispersal and extinction rates from fossils. Methods in Ecology and Evolution
[doi:10.1111/2041-210X.13845](https://onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.13845)

* See tutorial [5](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_5.md)


#### The age-dependent extinction models (ADE) are described here:

* Silvestro et al. (2019) A 450 million years long latitudinal gradient in age‐dependent extinction. Ecology Letters, [doi: 10.1111/ele.13441](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.13441).

* Hagen O., Andermann T., Quental T. B., Antonelli A., Silvestro D. (2017) Estimating Age-dependent Extinction: Contrasting Evidence from Fossils and Phylogenies. Systematic Biology, [doi: 10.1093/sysbio/syx082](https://academic.oup.com/sysbio/article/doi/10.1093/sysbio/syx082/4563320/Estimating-Agedependent-Extinction-Contrasting).

* For the Bayesian implementation see tutorial [4](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#age-dependent-extinction-ade-model). The ADE-NN model (using neural networks) is available [here](https://github.com/dsilvestro/PyRate/tree/master/ADE-NN)




#### The birth death chrono-species model (BDC) is described here:

* Silvestro D., Warnock R., Gavryushkina A., Stadler T. (2018) Closing the gap between palaeontological and neontological speciation and extinction rate estimates. Nature Communications, [doi: 10.1038/s41467-018-07622-y](https://www.nature.com/articles/s41467-018-07622-y).

* See tutorial [4](https://github.com/dsilvestro/PyRate/blob/master/tutorials/pyrate_tutorial_4.md#the-birth-death-chronospecies-bdc-model)



#### The fossilized birth-death-range model (FBDrange) is described here:

* Warnock R., Heath T. A., Stadler T. (2020) Assessing the impact of incomplete species sampling on estimates of speciation and extinction rates. Paleobiology, [doi: 10.1017/pab.2020.12](https://www.cambridge.org/core/journals/paleobiology/article/assessing-the-impact-of-incomplete-species-sampling-on-estimates-of-speciation-and-extinction-rates/8D82C01066E7E2A24F2A4A8ACAC2B69F).

* Stadler T., Gavryushkina A., Warnock R. C., Drummond A. J., Heath T. A. (2018). The fossilized birth-death model for the analysis of stratigraphic range data under different speciation modes. Journal of Theoretical Biology [doi: 10.1016/j.jtbi.2018.03.0](https://www.sciencedirect.com/science/article/pii/S002251931830119X).

* Tutorial in prep. 



#### The multi-trait extinction model (MTE) is described here:

* Pimiento C., Bacon C. D., Silvestro D., Handy A., Jaramillo C., Zizka A., Meyer X., Antonelli A. (2020) Selective extinction against redundant species buffers functional diversity. Proceedings of the Royal Society B [doi: 10.1098/rspb.2020.1162](https://royalsocietypublishing.org/doi/abs/10.1098/rspb.2020.1162).

* Tutorial in prep., example file [here](https://github.com/dsilvestro/PyRate/blob/master/example_files/Example_data_MTE.txt). 
