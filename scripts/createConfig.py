#=========================================================================================
# createConfig.py -----------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Authors: Mark Samuel Abbott ------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# Condense global tag code? Assosiate it with iterator for myYears? small dictionary?

# This script was written for Python 2.7.5.
# This file is meant to be called by submitCrab.sh, but could be run manually.
# This file calls in template config files for crab_*.py and run_*.py, and writes a version of the config file unique to each dataset (particle, year, datatype, mass) under the config directory for each year.
# This file calls the dictionary "datasetDict" from datasetDictionary.py, which is created by buildDict.py.
# This file takes user inputs to determine which config files to create.

import argparse # Assists parsing of arguments
import sys
import os
from datasetDictionary import datasetDict # Import the dictionary of sample files

# Define ANSI colors here for the output since I am extra:
def bluestr(string):
    return '\033[94m' + string + '\033[0m'
def redstr(string):
    return '\033[91m' + string + '\033[0m'
def pinkstr(string):
    return '\033[95m' + string + '\033[0m'
def cyanstr(string):
    return '\033[96m' + string + '\033[0m'
def grnstr(string):
    return '\033[92m' + string + '\033[0m'
def yelstr(string):
    return '\033[93m' + string + '\033[0m'

# This is just a quick work-around for a limitation of argparse text formatting, since argparse takes only one input for formatter_class.
class RawTextAndDescriptionFormatter(argparse.RawTextHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

# Parse options and arguments from the user; valid options no options or "-p ... -y ... -d ..."
parser = argparse.ArgumentParser(description=yelstr('This builds crab config files from templates. The output files are unique to each dataset. User options and arguments determine which config files are created. Running without any options or arguments creates all config files.'), 
                                formatter_class=RawTextAndDescriptionFormatter,
                                epilog=yelstr("Examples: \n") + "python createConfig.py " + yelstr("\nDo not use quotes when submitting multiple arguments for one option:\n") + "python createConfig.py " + cyanstr("-p all ") + grnstr("-y 2016_APV 2016 2018 ") + pinkstr("-d mc") )
parser.add_argument('-p', '--particle',  nargs='+', type=str,  
                    choices=["HH", "WW", "ZZ", "tt", "bb", "QCD", "all"],
                    help=cyanstr("Define which particle config files to create."),
                    metavar=cyanstr("all | {HH WW ZZ tt bb QCD}") )
parser.add_argument('-y', '--year', nargs='+', type=str, choices=["2016_APV", "2016", "2017", "2018", "all"], 
                    help=grnstr("Define which year config files to create."),
                    metavar=grnstr("all | {2016_APV 2016 2017 2018}") )
parser.add_argument('-d', '--datatype', nargs='+', type=str, choices=["mc", "data", "all"],
                    help=pinkstr("Define which datatype config files to create."),
                    metavar=pinkstr("all | {mc data}"))

optargs = parser.parse_args()
# Make sure the user passes valid inputs
if len(sys.argv) == 1: # Triggers if no options or arguments are passed
    print ( yelstr("Default behavior triggered. ") + bluestr("All ") + yelstr("crab config files for each ") + cyanstr("particle, ") + grnstr("year, ") + yelstr("and ") + pinkstr("datatype ") + yelstr("will be created.") )
elif not (optargs.particle and optargs.year and optargs.datatype): # Triggers if -p,-y, or -d are passed without all 3 of them being passed together
        print (redstr("INPUT ERROR: ") + yelstr("-p, -y, and -d must all be specified, or none can be specified. See help:\n") )
        parser.print_help()
        sys.exit(2)
else: # This makes sure that is "all" is passed for -p, -y, or -d, that it is the only argument passed. 
    for opt in optargs.__dict__: # Iterate over options
        if optargs.__dict__[opt]:
            for arg in range(len(optargs.__dict__[opt])): # Iterate over arguments for each option
                if optargs.__dict__[opt][arg] == "all" and len(optargs.__dict__[opt]) > 1:
                    print (redstr("INPUT ERROR: ") + yelstr("If 'all' is passed as an argument, it must be the only argument passed for that option.  See help:\n") )
                    parser.print_help()
                    sys.exit(2)


# At this point, the inputs are assured to be valid, and we can begin the main functions of this file.
# First, define the full list of valid arguments for each option, and then empty lists to fill with user input.
allParticles = ["HH", "WW", "ZZ", "tt", "bb", "QCD"]
allYears = ["2016_APV", "2016", "2017", "2018"]
allDatatypes = ["mc", "data"]
myParticles = []
myYears = []
myDatatypes = []

# These load the "my..." lists with the user chosen input:
if len(sys.argv) == 1 or optargs.particle[0] == "all":
    myParticles = allParticles
elif optargs.particle:
    myParticles = optargs.particle
if len(sys.argv) == 1 or optargs.year[0] == "all":
    myYears = allYears
elif optargs.year:
    myYears = optargs.year
if len(sys.argv) == 1 or optargs.datatype[0] == "all":
    myDatatypes = allDatatypes
elif optargs.datatype:
    myDatatypes = optargs.datatype

# print( cyanstr("Particle(s) selected: " + str(myParticles)) )
# print( grnstr("Year(s) selected: " + str(myYears)) )
# print( pinkstr("Datatype(s) selected: " + str(myDatatypes)) )

configtemplateFile = "templates/crab_template.py"
runtemplateFile = "templates/run_template.py"

# GlobalTags = ["106X_mcRun2_asymptotic_preVFP_v11","106X_mcRun2_asymptotic_v17", "106X_mc2017_realistic_v8", "106X_upgrade2018_realistic_v15_L1v1"]

print( yelstr("Creating config files...") )
for dat in myDatatypes:
    if dat == "data": continue # Skip data, not implemented yet

    for yr in myYears:
 
        crabPath = "submit" + yr + "/config" # Path to config directory
        if not os.path.exists(crabPath): os.makedirs(crabPath) # If config directory doesn't exist, create it

        # Load Global Tag:
        if yr == "2016_APV":
            GlobalTag = "106X_mcRun2_asymptotic_preVFP_v11"
        elif yr == "2016":
            GlobalTag = "106X_mcRun2_asymptotic_v17"
        elif yr == "2017":
            GlobalTag = "106X_mc2017_realistic_v8"
        elif yr == "2018":
            GlobalTag = "106X_upgrade2018_realistic_v15_L1v1"

        for part in myParticles:

            # Create the run config files for each year, for each particle
            runFile = crabPath + "/run_" + part + ".py"
            if os.path.exists(runFile): os.remove(runFile) # Delete old run config file
            runf = open(runFile, "w") # Open new run config file to write
            runtempf = open(runtemplateFile, "r") # Open template run config file to read 
            for line in runtempf:
                if "GLOBALTAGFLAG" in line:
                    runf.write( 'GT = "' + GlobalTag + '"\n' ) # Write global tag (unique by year)
                elif "PARTICLESTRINGFLAG" in line: 
                    runf.write( '\tjetType = cms.string("' + part[0] +'"),\n' ) # Write first letter of particle
                else: 
                    runf.write(line) # Copy the rest of the file
            runf.close
            runtempf.close

            # Create the crab config files for each year, particle, and mass point
            for k, v in datasetDict[dat][yr][part].items(): # Iterate through the dictionary (k) and [crabdir,dataset,file] (v)

                if part == "QCD":
                    configFile = crabPath + "/crab_" + part +"_Pt_" + k + ".py" # Create unique config file name, like "crab_QCD_Pt_470to600.py"
                else:
                    configFile = crabPath + "/crab_" + part +"_M_" + k + ".py" # Create unique config file name, like "crab_HH_M_500.py"

                if os.path.exists(configFile): os.remove(configFile) # Delete old crab config file
                conf = open(configFile, "w") # Open new crab config file to write
                contempf = open(configtemplateFile, "r") # Read in crab config template
                for line in contempf:
                    if "CRABDIRFLAG" in line: 
                        conf.write( 'config.General.requestName = "' + datasetDict[dat][yr][part][k][0] + '"\n' ) # Here the dictionary calls [k]th mass point's corresponding crab directory name (the [0])  
                    elif "RUNPARTICLEFLAG" in line:
                        conf.write( 'config.JobType.psetName = "run_' + part +'.py"\n' ) # Write the corresponding run config file to use 
                    elif "DATASETFLAG" in line:
                        conf.write( 'config.Data.inputDataset = "' + datasetDict[dat][yr][part][k][1] + '"\n' ) # Here the dictionary calls [k]th mass point's corresponding dataset name (the [1])  
                    else:
                        conf.write(line) # Copy the rest of the file
                conf.close
                contempf.close


        print( yelstr("Config files for ") + grnstr(crabPath) + yelstr(" complete!") )
print( yelstr("Finished creating config files.") )


# config.Data.inputDataset = '/QCD_Pt-15to7000_TuneCP5_Flat_13TeV_pythia8/RunIISummer16MiniAODv3-PUFlat0to70_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM'
# #config.Data.splitting = 'FileBased' #auto for 2016/2018, file for 2017...file for everything else
