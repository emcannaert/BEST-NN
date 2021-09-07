#=========================================================================================
# buildDict.py ---------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script was written for Python 2.7.5.
# This python script builds a nested dictionary out of DAS datasets filled by fillSamples.sh, and this dictionary is then used by ####CRABPYTHON####.
# The user can specify the keys particle, year, datatype, and masspoint to get any unique entry. The values returned are the crab directory name, the dataset name, and a relevant file from the datset. 


import os
# Define lists of variables:
allParticles = ["HH", "WW", "ZZ", "tt", "bb", "QCD"]
allYears = ["2016_APV", "2016", "2017", "2018"]
allDatatypes = ["mc", "data"]

# Define the initial dictionary that holds all others:
d = {}
# Now we fill this dictionary with dictionaries, created a nested dictionary with 4 levels:
for dat in allDatatypes:
    if dat == "data": continue # Skip data, not implemented yet
    # print(dat)
    d[dat] = {} # Creates 2 empty dictionaries for the keys ["mc", "data"]
    for yr in allYears:
        # print(yr)
        d[dat][yr] = {} # Creates 4 empty dictionaries with keys ["2016_APV", "2016", "2017", "2018"], for both previous dictionaries (8 new dictionaries total)
        sampleFile = "../samples/" + dat + "_" + yr + ".txt" 
        with open(sampleFile) as f: # Opens sample file, code will implicitly close file when done with loop
            # print(sampleFile)
            for line in f:
                if line[0] == "#": # These mark each particle section, like "#particle"
                    part = line.strip()[1:] # Stores the particle without the '#' character
                    # print(part)
                    d[dat][yr][part] = {} # Creates 6 empty dictionaries with keys ["HH", "WW", "ZZ", "tt", "bb", "QCD"], for all previous dictionaries (48 new dictionaries total)
                else:
                    mass, crab, dataset = line.split(',') # Reads in comma seperated values
                    # print(mass,crab,dataset)
                    # print("load dict")
                    # Here we actuallly load our data into the dictionary, with the innermost layer using mass for keys and returned crab directory, dataset, and file as values.
                    d[dat][yr][part][mass] = [crab,dataset.strip()] # The strip is needed to get rid of the \n at the end of dataset.
                    # print(d[dat][yr][part][mass])

if os.path.exists("dict.py"):
        os.remove("dict.py")
file = open("dict.py", "w")
file.write("d = " + str(d))
file.close

# To use this dictionary, simply add "import PATH/buildDict.py"

# print("done filling dicts")
# print(d)
# print("\nBUFFFFFFFFFFFFFFER\n")
# print(d["mc"]["2016_APV"])
# print("\nBUFFFFFFFFFFFFFFER\n")
# print(d["mc"]["2016_APV"]["HH"])
# print("\nBUFFFFFFFFFFFFFFER\n")
# print(d["mc"]["2016_APV"]["HH"]["5000"])
# print("\nBUFFFFFFFFFFFFFFER\n")

# print(d["mc"]["2016_APV"]["HH"].items())
# for dat in allDatatypes:
#     if dat == "data": continue # Skip data, not implemented yet
#     print(dat)
#     for yr in allYears:
#         print(yr)
#         for par in allParticles:
#             print(par)
#             print()
#             for k, v in d[dat][yr][par].items():
#                 print(k)
#                 print(v)
# print(d)

