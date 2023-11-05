#!/bin/bash
#=========================================================================================
# submitLocal.sh -------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script will cmsRun BEST jobs locally

# Define ANSI colors here for the output since I am extra:
CYAN='\033[96m' # Light Cyan
YEL='\033[93m' # Yellow
NC='\033[0m' # No Color

declare -a allParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")


# Submit jobs in current directory:
for part in ${allParticles[*]}; do # Loop over particles
    listOfScripts=run*$part*.py
    echo -e "${YEL}Running ${CYAN}$part${NC}..."

    for f in $listOfScripts; do
        cmsRun $f
    done
done
