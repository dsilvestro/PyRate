
source("/Users/shelleywang/Documents/PU RSII/PyRate/pyrate_utilities.r")

extant_dogs = c("Canis rufus","Canis lupus","Canis aureus","Canis latrans","Canis mesomelas","Canis anthus","Pseudalopex gymnocercus","Canis adustus","Canis familiaris")

extract.ages.pbdb(file = "/Users/shelleywang/Documents/PU RSII/PyRate/tutorials/Canis_self_example/Canis_pbdb_data.csv", extant_species = extant_dogs)
# Did not set a replicates argument here, so further commands do not contain the -j flag

# *_TaxonList.txt, *_PyRate.py, *.txt created in the working directory

# Terminal Data Summary: python PyRate.py 'tutorials/Canis_self_example/Canis_pbdb_data.csv' -data_info | tee tutorials/Canis_self_example/Canis_data_summary.txt
# Terminal Taxon Typos Check: python PyRate.py -check_names Canis_pbdb_data_TaxonList.txt

# Terminal Run PyRate: python PyRate.py tutorials/Canis_self_example/Canis_pbdb_data.csv.py and all flags neede