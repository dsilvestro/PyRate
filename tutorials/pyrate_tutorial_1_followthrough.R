source("pyrate_utilities.r")

extant_dogs = c("Canis rufus","Canis lupus","Canis aureus","Canis latrans","Canis mesomelas","Canis anthus","Pseudalopex gymnocercus","Canis adustus","Canis familiaris")
print(extant_dogs)
extract.ages.pbdb(file = "example_files/Canis_example/Canis_pbdb_data.csv", extant_species = extant_dogs)
# This should've created two files 