#=========================================================================================
# buildDict.py ---------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script was written for Python 2.7.5.
# This script is meant to be called by fillSamples.sh, but can be run manually. It is designed to be ran from the BEST/preprocess/crab directory, but could be run elsewhere if the path for "sampleFile" and "outputFile" is edited.
# This python script builds a nested dictionary out of DAS datasets filled by fillSamples.sh, and this dictionary is then used by createConfig.py.
# The user can specify the keys particle, year, datatype, and masspoint to get any unique entry. The values returned are the crab directory name, the dataset name, and the file with the greatest amount of events from the datset. 
# This file does not take any options--if sample input data is missing (year, particle, datatype), the output dictionary will simply not contain that information.

import os
def yelstr(string): 
    return '\033[93m' + string + '\033[0m'

# Define lists of variables:
allParticles = ["HH", "WW", "ZZ", "tt", "bb", "QCD"]
allYears = ["2016_APV", "2016", "2017", "2018"]
allDatatypes = ["mc", "data"]

print( yelstr("Building dataset dictionary...") )

# Define the initial dictionary that holds all others:
datasetDict = {}
# Now we fill this dictionary with dictionaries, created a nested dictionary with 4 levels:
for dat in allDatatypes:
    if dat == "data": continue # Skip data, not implemented yet
    datasetDict[dat] = {} # Creates 2 empty dictionaries for the keys ["mc", "data"]
    for yr in allYears:
        datasetDict[dat][yr] = {} # Creates 4 empty dictionaries with keys ["2016_APV", "2016", "2017", "2018"], for both previous dictionaries (8 new dictionaries total)
        sampleFile = "../../samples/" + dat + "_" + yr + ".txt" 
        with open(sampleFile) as f: # Opens sample file, code will implicitly close file when done with loop
            next(f) # Skip the first line (header)
            for line in f:
                if line[0] == "#": # These mark each particle section, like "#particle"
                    part = line.strip()[1:] # Stores the particle without the '#' character
                    datasetDict[dat][yr][part] = {} # Creates 6 empty dictionaries with keys ["HH", "WW", "ZZ", "tt", "bb", "QCD"], for all previous dictionaries (48 new dictionaries total)
                else:
                    mass, crab, dataset, datasetFile = line.split(',') # Reads in comma seperated values
                    # Here we actuallly load our data into the dictionary, with the innermost layer using mass for keys and returned crab directory, dataset, and file as values.
                    datasetDict[dat][yr][part][mass] = [crab,dataset,datasetFile.strip()] # The strip is needed to get rid of the \n at the end of datasetFile, since it is the last entry in each line.

# Write the raw dictionary to a python file, to be imported in other files:
outputFile = "datasetDictionary.py"
outPath = "../../scripts/" + outputFile
if os.path.exists(outPath): os.remove(outPath)
file = open(outPath, "w")
file.write("datasetDict = " + str(datasetDict))
file.close

print( yelstr("Done building dataset dictionary! Check it out at: " + outputFile) )