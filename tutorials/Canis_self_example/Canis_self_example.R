
source("../../pyrate_utilities.r")

extant_dogs = c("Canis rufus","Canis lupus","Canis aureus","Canis latrans","Canis mesomelas","Canis anthus","Pseudalopex gymnocercus","Canis adustus","Canis familiaris")

extract.ages.pbdb(file = "Canis_pbdb_data.csv", extant_species = extant_dogs, replicates=2)
# Set replicates = 2, so future commands will need -j flag

# *_TaxonList.txt, *_PyRate.py, *.txt created in the working directory

# TERMINAL COMMANDS FOLLOW:
# Data Summary: python PyRate.py 'tutorials/Canis_self_example/Canis_pbdb_data.csv' -data_info | tee tutorials/Canis_self_example/Canis_data_summary.txt
# Canis_data_summary.txt

# Taxon Typos Check: python PyRate.py -check_names Canis_pbdb_data_TaxonList.txt
# *_TaxonList_scores.txt created in working directory

# Run PyRate (Gamma TPP + default BD Model): python PyRate.py tutorials/Canis_self_example/Canis_pbdb_data.csv.py 

# Model Likelihood Test: py PyRate.py .\tutorials\Canis_self_example\Canis_pbdb_data_PyRate.py -qShift .\tutorials\Canis_self_example\epochs_q.txt -PPmodeltest
# Outputted TPP with lowest AIC score. It is recommended that -mG (Gamma model) is added always 

# MCMC Analysis Set Up Replicate 1: py PyRate.py .\tutorials\Canis_self_example\Canis_pbdb_data_PyRate.py -qShift .\tutorials\Canis_self_example\epochs_q.txt -mG -j 1
# TPP + Gamma model, replicate 1, RJMCMC, default MCMC -n (iterations) and -s (samples)
# Output: *_1_Grj_sum.txt, *_1_Grj_sp_rates.log, *_1_Grj_ex_rates.log, *_Grj_1_mcmc.log

# MCMC Analysis Set Up Replicate 2: py PyRate.py .\tutorials\Canis_self_example\Canis_pbdb_data_PyRate.py -qShift .\tutorials\Canis_self_example\epochs_q.txt -mG -j 2
# TPP + Gamma model, replicate 2, RJMCMC, default MCMC -n (iterations) and -s (samples)
# Output: *_2_Grj_sum.txt, *_2_Grj_sp_rates.log, *_2_Grj_ex_rates.log, *_Grj_2_mcmc.log


# Add all MCMC log files to Tracer, examine burn-in samples
# Used default MCMC -n 10 mill and -s 1000, decided on a burn-in of about 1 mill iterations --> 1 mill/s --> 1000 burn-in samples

# BD Model Sampling Frequencies for Replicate 1: python PyRate.py -mProb "\tutorials\Canis_self_example\pyrate_mcmc_logs\Canis_pbdb_data_1_Grj_mcmc.log" -b 1000
# BD Model Sampling Frequencies for Replicate 2: python PyRate.py -mProb "\tutorials\Canis_self_example\pyrate_mcmc_logs\Canis_pbdb_data_2_Grj_mcmc.log" -b 1000
# Output: A table showing which # of rate shifts has the highest probability
#For both replicates, the most likely configuration was 1 Speciation rate shift, 2 Extinction rate shifts

# RTT plot replicate 1 using mcmc.log: python PyRate.py -mProb tutorials/Canis_self_example/pyrate_mcmc_logs/Canis_pbdb_data_1_Grj_mcmc.log -b 1000
# RTT plot replicate 2: python PyRate.py -mProb tutorials/Canis_self_example/pyrate_mcmc_logs/Canis_pbdb_data_2_Grj_mcmc.log -b 1000


#after py PyRate.py .\tutorials\Canis_self_example\Canis_pbdb_data_PyRate.py -qShift .\tutorials\Canis_self_example\epochs_q.txt -mG -j 1 -log_marginal_rates 1
#which is mcmc analysis set up for replicate 1, but with the addition of a -log_marginal_rates 1 tag to output the *_marginal_rates.log file needed
#for an RTT plot in section 5.C., check whether *_sp_rates and *_ex_rates has been update to July 24th (b/c that would mean the new flag outputs ALL output files, in addition to the *_marginal_rates.log)