Fossil occurrence data from [Pimiento et al. 2017](https://doi.org/10.1038/s41559-017-0223-6).
The data are also used by [Silvestro et al. 2019](https://doi.org/10.1017/pab.2019.23) and analyzed using the following command:

```python PyRate.py occs.py -qShift -filter_taxa mammals.txt -qShift epochs_q.txt -mG -pP 1.5 0```
      
The original dataset contained other marine megafauna organisms whereas here we decided to focus on mammals only, we used the command `-filter_taxa mammals.txt` to provide a list of mammalian taxa that we want to include in the analysis (whereas all other lineages are dropped).
Additional commands: `-qShift` specifies that preservation is modeled by a TPP process with independent rates for each epoch, `-mG` specifies that the TPP model should be coupled by a Gamma model of rate heterogeneity across lineages, and `-pP 1.5 0` specifies the shape and rate parameters of the gamma prior on the preservation rates. By setting the rate parameter to 0 we define the parameter as unknown, meaning that PyRate will estimate it after assigning it a hyper-prior.
