
source("/Users/shelleywang/Documents/PU RSII/PyRate/pyrate_utilities.r")

extant_dogs = c("Canis rufus","Canis lupus","Canis aureus","Canis latrans","Canis mesomelas","Canis anthus","Pseudalopex gymnocercus","Canis adustus","Canis familiaris")

extract.ages.pbdb(file = "/Users/shelleywang/Documents/PU RSII/PyRate/tutorials/Canis_self_example/Canis_pbdb_data.csv", extant_species = extant_dogs)
# Did not set a replicates argument here, so further commands do not c