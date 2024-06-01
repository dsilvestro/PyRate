# GENERATE INPUT FILE OPTION 1
# Load pyrate_utilities.r
source("../pyrate_utilities.r")

# Defining a vector of extant species
extant_dogs = c("Canis rufus","Canis lupus","Canis aureus","Canis latrans","Canis mesomelas","Canis anthus","Pseudalopex gymnocercus","Canis adustus","Canis familiaris")

# Extract fossil occurrence data from raw PBDB table, save it in a PyRate-compatible.csv
extract.ages.pbdb(file= "../example_files/Canis_example/Canis_pbdb_data.csv",extant_species=extant_dogs)

# Two files should've been created by ^: 
#*_SpeciesList.txt (list of all species in the data) and *_PyRate.py (data formatted for PyRate analysis)

# Get Summary Statistics of PyRate Input file (*_PyRate.py)
# Open TERMINAL, navigate to the PyRate folder, and run the following command:
# python PyRate.py 'example_files/Canis_example/Canis_pbdb_data_PyRate.py' -data_info

# Check Name Spellings (*_SpeciesList.txt)
# Open TERMINAL, run this, and it should return a table with possible misspellings
# python PyRate.py  -check_names example_files/Canis_example/Canis_pbdb_data_TaxonList.txt -data_info
# Rank 1 and 0 = most likely misspellings, Rank 2 and 3 = most likely truly different names
# Fix the names in the dataset if needed

# Estimation of speciation and extinction rates through time
# Set up the preservation model
os.system("python PyRate.py '.../Canis_pbdb_data_PyRate.py' -mHPP")  # Homogeneous Poisson process (HPP)
os.system("python PyRate.py '.../Canis_pbdb_data_PyRate.py' -qShift '.../epochs_q.txt' -mG")  # Time-variable Poisson process (TPP) with Gamma model
os.system("python PyRate.py '.../Canis_pbdb_data_PyRate.py' -mG")  # Non-homogeneous Poisson process (NHPP) with Gamma model

# Run the analysis
os.system("python PyRate.py '.../Canis_pbdb_data_PyRate.py' -A 2 -j 1")  # Run BDMCMC algorithm on the first replicate
os.system("python PyRate.py '.../Canis_pbdb_data_PyRate.py' -A 4 -j 2")  # Run RJMCMC algorithm on the second replicate

# Summarize the results
os.system("python PyRate.py -mProb '.../Canis_pbdb_data_mcmc.log' -b 200")  # Calculate sampling frequencies of birth-death models
os.system("python PyRate.py -plot '.../Canis_pbdb_data_marginal_rates.log' -b 200")  # Generate rates-through-time plots
os.system("python PyRate.py -plotQ '.../Canis_pbdb_data_mcmc.log' -qShift 'epochs.txt' -b 100")  # Plot preservation rates through time

# Speciation and extinction rates within fixed time bins
os.system("python PyRate.py '.../Canis_pbdb_data_PyRate.py' -fixShift '.../epochs.txt'")

# Set fixed shifts at the boundaries, while searching for rate shifts between them
os.system("python PyRate.py <data_set> -A 4 -edgeShift 18 2")