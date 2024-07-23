
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

# Run PyRate: python PyRate.py tutorials/Canis_self_example/Canis_pbdb_data.csv.py 

# Model Likelihood Test: py PyRate.py .\tutorials\Canis_self_example\Canis_pbdb_data_PyRate.py -qShift .\tutorials\Canis_self_example\epochs_q.txt -PPmodeltest
# Outputted TPP with lowest AIC score. It is recommended that -mG (Gamma model) is added always 

# MCMC Analysis Set Up: