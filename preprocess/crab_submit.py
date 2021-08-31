#=========================================================================================
# crab_submit.py -------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Authors: Mark Samuel Abbott ------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This file submits jobs to crab by calling "run_crab.py", and lives in the directory BEST/preprocess.
# The datasets are pulled from BEST/samples, which is filled by running fillSamples.sh.
# This file takes user inputs to determine which files to submit to crab.

######################################### NOTES TO SELF ############################
# Check with johan
# figure out best way to submit crab (over and over? all at once? script? python?)

from WMCore.Configuration import Configuration
import argparse
import sys

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

# Parse options and arguments from the user; valid options are "-a" or "-p ... -y ... -d ..."
parser = argparse.ArgumentParser(description=yelstr('This submits jobs to CRAB by calling "run_crab.py". User options and arguments determine which jobs are submitted.'), 
                                formatter_class=RawTextAndDescriptionFormatter,
                                epilog=yelstr("Examples: \n") + "python crab_submit.py " + bluestr("-a\n") + yelstr("Do not use quotes when submitting multiple arguments for one option:\n") + "python crab_submit.py " + cyanstr("-p HH bb QCD ") + grnstr("-y all ") + pinkstr("-d mc") )
parser.add_argument('-a', '--all', action="store_true", 
                    help=bluestr("Submit all crab jobs \n(mutually exclusive with ") + cyanstr("-p,") + grnstr(" -y,") + bluestr(" and") + pinkstr(" -d") + bluestr(")") )
parser.add_argument('-p', '--particle', type=str, nargs='+',  
                    choices=["HH", "WW", "ZZ", "tt", "bb", "QCD", "all"],
                    help=cyanstr("Define which particle datasets to submit \n(mutually exclusive with ") + bluestr("-a") + cyanstr(")"),
                    metavar=cyanstr("all | {HH WW ZZ tt bb QCD}") )
parser.add_argument('-y', '--year', nargs='+', type=str, choices=["2015", "2016", "2017", "2018", "all"], 
                    help=grnstr("Define which year datasets to submit \n(mutually exclusive with ") + bluestr("-a") + grnstr(")"),
                    metavar=grnstr("all | {2015 2016 2017 2018}") )
parser.add_argument('-d', '--datatype', nargs='+', type=str, choices=["mc", "data", "all"],
                    help=pinkstr("Define which datatype datasets to submit \n(mutually exclusive with ") + bluestr("-a") + pinkstr(")"),
                    metavar=pinkstr("all | {mc data}"))

optargs = parser.parse_args()

# Make sure the user passes valid inputs
if optargs.all and (optargs.particle or optargs.year or optargs.datatype):
    print (redstr("INPUT ERROR: ") + yelstr("-a and {-p|-y|-d} are mutually exclusive. See help:\n"))
    parser.print_help()
    sys.exit(2)
elif not optargs.all:
    if not (optargs.particle and optargs.year and optargs.datatype):
        print (redstr("INPUT ERROR: ") + yelstr("-p, -y, and -d must all be specified. See help:\n") )
        parser.print_help()
        sys.exit(2)
    else:
        for opt in optargs.__dict__:
            if optargs.__dict__[opt]:
                for arg in range(len(optargs.__dict__[opt])):
                    if optargs.__dict__[opt][arg] == "all" and len(optargs.__dict__[opt]) > 1: #set the keys. probably need to set up a dest for each thing so i can overwrite it
                        print (redstr("INPUT ERROR: ") + yelstr("If 'all' is passed as an argument, it must be the only argument passed for that option.  See help:\n") )
                        parser.print_help()
                        sys.exit(2)

print(optargs)
# print(bluestr(str(optargs.all)))
# print(cyanstr(str(optargs.particle)))
# print(grnstr(str(optargs.year)))
# print(pinkstr(str(optargs.datatype)))

# At this point, the inputs are assured to be valid, and we can begin the main functions of this file.

# First, define the full list of valid arguments for each option, and then empty lists to fill with user input.
allParticles = ["HH", "WW", "ZZ", "tt", "bb", "QCD"]
allYears = ["2015", "2016", "2017", "2018"]
allDatatypes = ["mc", "data"]
myParticles = []
myYears = []
myDatatypes = []

# These load the my"..." lists with the user chosen input
if optargs.all or optargs.particle[0] == "all":
    myParticles = allParticles
elif optargs.particle:
    myParticles = optargs.particle
if optargs.all or optargs.year[0] == "all":
    myYears = allYears
elif optargs.year:
    myYears = optargs.year
if optargs.all or optargs.datatype[0] == "all":
    myDatatypes = allDatatypes
elif optargs.datatype:
    myDatatypes = optargs.datatype

print(cyanstr(str(myParticles)))
print(grnstr(str(myYears)))
print(pinkstr(str(myDatatypes)))

config = Configuration()
config.section_("General")


# for dat in myDatatypes:
#     print pinkstr(dat)
#     if dat == "data": #data is not implemented yet
#         continue
#     for yr in myYears:
#         print grnstr(yr)
#         filename =  dat+"_"+yr+".txt"
#         print "Opening " + filename + "..."
#         for line in open(filename):
#             #blah
#         for part in myParticles:
#             print cyanstr(part)



#turn this into some sort of array, loop; 
# config.General.requestName = 'ZprimeBB_2TeV_trees'
# config.General.requestName = 'GravitonHH_2TeV_trees'
# config.General.requestName = 'QCD_Flat_Pt_trees'
# config.General.requestName = 'ZprimeTT_2TeV_trees'
# config.General.requestName = 'ZprimeWW_2TeV_trees'
# config.General.requestName = 'RadionZZ_5TeV_trees'

# config.General.workArea = 'CrabBEST'
# config.General.transferLogs = True

# config.section_("JobType")
# config.JobType.pluginName = 'Analysis'

# config.JobType.psetName = 'run_crab.py'
# # config.JobType.psetName = 'run_bb.py'
# # config.JobType.psetName = 'run_HH.py'
# # config.JobType.psetName = 'run_QCD.py'
# # config.JobType.psetName = 'run_tt.py'
# # config.JobType.psetName = 'run_WW.py'
# # config.JobType.psetName = 'run_ZZ.py'


# #config.JobType.inputFiles = ['TMVARegression_MLP.weights.xml']
# config.JobType.outputFiles = ['BESTInputs.root']
# #config.JobType.allowUndistributedCMSSW = True

# config.section_("Data")

# #Updat this to call samples files, based on dasPart and dasYear
# config.Data.inputDataset = '/ZprimeToBB_narrow_M-2000_TuneCP5_13TeV-madgraph-pythia8/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM'
# config.Data.inputDataset = '/BulkGravTohhTohbbhbb_narrow_M-2000_13TeV-madgraph/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3_ext1-v1/MINIAODSIM'
# config.Data.inputDataset = '/ZprimeToTT_M-2000_W-20_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v2/MINIAODSIM'
# config.Data.inputDataset = '/ZprimeToWW_narrow_M-2000_13TeV-madgraph/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM'
# config.Data.inputDataset = '/RadionToZZ_narrow_M-5000_TuneCUETP8M1_13TeV-madgraph-pythia8/RunIISummer16MiniAODv3-PUMoriond17_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM'

# config.Data.splitting = 'FileBased'
# config.Data.unitsPerJob = 1

# config.Data.inputDataset = '/QCD_Pt-15to7000_TuneCP5_Flat_13TeV_pythia8/RunIISummer16MiniAODv3-PUFlat0to70_94X_mcRun2_asymptotic_v3-v1/MINIAODSIM'
# config.Data.splitting = 'Automatic'
# #config.Data.splitting = 'FileBased' #auto for 2016/2018, file for 2017...file for everything else
# #config.Data.unitsPerJob = 10

# config.Data.ignoreLocality = True
# config.Data.publication = False
# # This string is used to construct the output dataset name

# config.section_("Site")
# config.Site.storageSite = 'T3_US_FNALLPC'
# config.Site.whitelist = ['T2_US_*']


# This script takes arguments (sample keys) from the user and submits the appropriate jobs to crab.
# The script lives in the scripts directory and the symbolic links in each of the submit201X directories should be executed within their respective directories.


# # Extract keys from argument(s)
# declare -a myKeys
# if [ $1 == "all" ]; then
#     echo "Submitting all samples"
#     myKeys=("HH" "WW" "ZZ" "tt" "bb" "QCD")
# else
#     for arg; do
# 	myKeys+=($arg)
#     done
# fi
# echo "Submitting samples: ${myKeys[@]}"

# # Check if logFiles directory exists, else make one
# if [[ -d "logFiles" ]]
# then
#     echo "logFiles exists on your filesystem."
# else
#     mkdir logFiles
#     echo "Created logFiles directory."
# fi

# # Find all crab scripts of each key and submit them in parallel
# for myKey in ${myKeys[@]}; do
#     echo "Finding submition scripts for $myKey"
#     listOfScripts=crab*$myKey*.py
#     for f in $listOfScripts; do
# 	crabName=$(echo $f | cut -d'.' -f 1)
# 	crab submit $f >> logFiles/$crabName.txt &
#     done
# done
