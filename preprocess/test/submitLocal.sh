#!/bin/bash
#=========================================================================================
# submitCrab.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script takes options/arguments from the user and submits the appropriate jobs to crab. 
# This script will also call createConfig.py to generate the crab config files to submit.
# The script lives in the scripts directory and the symbolic links in each of the submit201X directories should be executed within their respective directories.

# ssh -XY msabbott@cmslpc114.fnal.gov
# screen
# bash -l
# cd nobackup/abbott/CMSSW_10_6_27/src/
# cmsenv
# scram b -j8
# cd BEST/preprocess/test/mid
# cmsRun run_

# 115 tar

# Need to rename high top to 4500
#   low mlf mid high cmslpc###
# H         115 
# W         114    
# Z         126
# t         118
# b 118     122 122
# Q     112 112  

# Define ANSI colors here for the output since I am extra:
RED='\033[91m' # Red
CYAN='\033[96m' # Light Cyan
BLUE='\033[94m' # Blue
PURP='\033[35m' # Light Purple
GRN='\033[92m' # Light Green
YEL='\033[93m' # Yellow
NC='\033[0m' # No Color
# This alias makes the script simpler, as '-e' is needed to print color. This is undone by 'unalias' at the end of the code.
shopt -s expand_aliases
alias echo='echo -e'

# Declare the full list of valid arguments for each option
declare -a allParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")
declare -a allYears=("2016_APV" "2016" "2017" "2018")
declare -a allDatatypes=("mc" "data")

################# At this point, the code unique to this file begins: 

# Use loop structure below to submit crab jobs. 

for part in ${allParticles[*]}; do # Loop over particles
    listOfScripts=run*$part*.py
    echo "${YEL}Running ${CYAN}$part${NC}..."
    for f in $listOfScripts; do
        cmsRun $f
    done
done

#Undo the alias used for this script
unalias echo