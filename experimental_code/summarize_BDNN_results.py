import numpy as np
np.set_printoptions(suppress= 1, precision=3)
import os, csv, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from plot_BDNN import predict_rates_per_species, parse_sum_txt_file

# paths and file names 
wd = "bdnn_analysis"
wd_logfiles = "bdnn_analysis/pyrate_mcmc_logs"


### EXAMPLE MODEL 2
species_trait_file= os.path.join(wd,  "filename_traits_taxa.txt")
logfile = os.path.join(wd_logfiles,   "filename_m2_G_BDS_BDNN3_mcmc.log")
sumfile = os.path.join(wd_logfiles,   "filename_m2_G_BDS_BDNN3_sum.txt")
model_name = "_m2" # used for output files

# get model settings
times_of_shift, mid_points, time_as_trait = parse_sum_txt_file(sumfile)

# Get species-specific rates
species_traits = pd.read_csv(species_trait_file, delimiter="\t")
species_traits = species_traits.rename(columns={'Species': 'Taxon_name'}) # rename taxon column


res = predict_rates_per_species(logfile=logfile, 
                                species_trait_file=None,
                                trait_tbl=species_traits,
                                wd=wd, 
                                time_range = mid_points,
                                burnin = 0.2,
                                fixShift = times_of_shift,
                                rescale_time = time_as_trait,
                                return_post_sample = True,
                                out=model_name)


### EXAMPLE MODEL 3
species_trait_file= os.path.join(wd,  "filename_traits_taxa.txt")
logfile = os.path.join(wd_logfiles,   "filename_m3_G_BDS_BDNN3T_mcmc.log")
sumfile = os.path.join(wd_logfiles,   "filename_m3_G_BDS_BDNN3T_sum.txt")
model_name = "_m3" # used for output files

# get model settings
times_of_shift, mid_points, time_as_trait = parse_sum_txt_file(sumfile)

# Get species-specific rates
species_traits = pd.read_csv(species_trait_file, delimiter="\t")
species_traits = species_traits.rename(columns={'Species': 'Taxon_name'}) # rename taxon column


res = predict_rates_per_species(logfile=logfile, 
                                species_trait_file=None,
                                trait_tbl=species_traits,
                                wd=wd, 
                                time_range = mid_points,
                                burnin = 0.2,
                                fixShift = times_of_shift,
                                rescale_time = time_as_trait,
                                return_post_sample = True,
                                out=model_name)



# save rates posterior samples 
# >>> res.shape
# >>> (n_time_bins, 2[sp/ex rates], n_posterior_samples, n_species_in_species_trait_file)

np.save(os.path.join(wd,"species_specific_rates%s.npy" % model_name), res)

