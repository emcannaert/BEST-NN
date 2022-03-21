#!/bin/bash
#=========================================================================================
# crabSubmit.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott  ---------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script submits user chosen jobs and submits them to crab. 
# This script will also call buildConfig.py to generate the crab config files to submit.
# The script lives in the scripts directory and the symbolic links in each of the submit201X directories should be executed within their respective directories.

#==================================================================================
# Setup ///////////////////////////////////////////////////////////////////////////
#==================================================================================

# Declare particles, years, and datatypes to submit. 
# These need to be edited manually, as passing an argument to ./crabSubmit.sh breaks the 'crab submit ...' command later.
declare -a myParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")
# declare -a myParticles=("QCD")
declare -a myYears=("2016_APV" "2016" "2017" "2018")
# declare -a myYears=("2017")
declare -a myDatatypes=("mc" "data")

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


# This is where the options and arguments are parsed in.
if [[ $# == 0 ]]; then # Default case, sets up to submit everything.
    echo "Default behavior triggered. ${BLUE}All${NC} crab jobs for each ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype${NC}, will be submitted."
    myParticles=${allParticles[*]}
    myYears=${allYears[*]}
    myDatatypes=${allDatatypes[*]}
else # Specific cases, sets up to submit specfic crab jobs.
    while getopts :p:y:d: opt; do
        case $opt in
            p)  # The -p option, for particles. Either fills "all" or the specificly chosen arguments.
                if [[ $OPTARG == "all" ]]; then
                    myParticles=${allParticles[*]}
                else 
                    for part in $OPTARG; do    
                        if [[ ${allParticles[*]} =~ $part ]]; then # Check for valid arguments, then fills array.
                            myParticles+=($part)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${CYAN}$opt${NC}: $part"
                            echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${CYAN}${allParticles[*]}${NC}"
                            echo "${YEL}Run script without any options to see usage:${NC} ./crabSubmit.sh"
                            echo "${RED}Exiting without submitting jobs...${NC}"
                            exit 1
                        fi
                    done
                fi
            ;;
            y)  # The -y option, for years. Either fills "all" or the specificly chosen arguments.
                if [[ $OPTARG == "all" ]]; then
                    myYears=${allYears[*]}
                else
                    for yr in $OPTARG; do
                        if [[ ${allYears[*]} =~ $yr ]]; then # Check for valid arguments, then fills array.
                            myYears+=($yr)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${GRN}$opt${NC}: $yr"
                            echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${GRN}${allYears[*]}${NC}"
                            echo "${YEL}Run script without any options to see usage:${NC} ./crabSubmit.sh"
                            echo "${RED}Exiting without submitting jobs...${NC}"
                            exit 1
                        fi
                    done
                fi
            ;;
            d)  # The -d option, for datatype. Either fills "all" or the specificly chosen arguments.
                if [[ $OPTARG == "all" ]]; then
                    myDatatypes=${allDatatypes[*]}
                else
                    for dat in $OPTARG; do
                        if [[ ${allDatatypes[*]} =~ $dat ]]; then # Check for valid arguments, then fills array.
                            myDatatypes+=($dat)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${PURP}$opt${NC}: $dat"
                            echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${PURP}${allDatatypes[*]}${NC}"
                            echo "${YEL}Run script without any options to see usage:${NC} ./crabSubmit.sh"
                            echo "${RED}Exiting without creating samples...${NC}"
                            exit 1
                        fi
                    done
                fi
            ;;
            \?) # Catches invalid options
                echo "${YEL}Error:${NC} Invalid option: -$OPTARG"
                echo "${YEL}Run script without any options to see usage:${NC} ./crabSubmit.sh"
                echo "${RED}Exiting without creating samples...${NC}"
                exit 1
            ;;
            :)  # Catches options missing arguments
                echo "${YEL}Error:${NC} Option -$OPTARG requires an argument."
                echo "${YEL}Run script without any options to see usage:${NC} ./crabSubmit.sh"
                echo "${RED}Exiting without creating samples...${NC}"
                exit 1
            ;;
        esac
    done
fi

echo "${CYAN}Particle(s)${NC} selected: ${CYAN}${myParticles[*]}${NC}"
echo "${GRN}Year(s)${NC} selected: ${GRN}${myYears[*]}${NC}"
echo "${PURP}Datatype(s)${NC} selected: ${PURP}${myDatatypes[*]}${NC}"
echo
echo "${BLUE}Submitting crab jobs. Checking voms cms proxy...${NC}"

# This checks for a voms cms proxy that will last longer than 60 minutes, and has the user create a new one if not
if [[ ! $(voms-proxy-info -timeleft) > 3600 ]] || [[ $(voms-proxy-info -vo) != "cms" ]]; then
    echo "${YEL}Error: Proxy either doesn't exist or will expire soon. Initializing new proxy...${NC}"
    voms-proxy-init --valid 192:00 -voms cms
    if [[ ! $(voms-proxy-info -timeleft) > 3600 ]] || [[ $(voms-proxy-info -vo) != "cms" ]]; then exit 1; fi # Exit if user failed to create proxy
fi
echo "${GRN}Valid proxy confirmed!${NC}"

################# At this point, the code unique to this file begins: 

# Call buildConfig.py to generate the config files:
python buildConfig.py -p ${myParticles[*]} -y ${myYears[*]} -d ${myDatatypes[*]}

eval `scramv1 runtime -sh` # This is the alias to cmsenv
source /cvmfs/cms.cern.ch/crab3/crab.sh # Source crab

#==================================================================================
# Submit Crab /////////////////////////////////////////////////////////////////////
#==================================================================================

# Use loop structure below to submit crab jobs. 
for dat in ${myDatatypes[*]}; do # Loop over mc and data
    if [[ $dat == "data" ]] ; then continue; fi #skips the loop for data, not implemented yet

    for yr in ${myYears[*]}; do # Loop over years
        yearDir="submit$yr"
        mkdir -p $yearDir # Make sure $yearDir exists
        echo "${YEL}Entering $yearDir...${NC}"
        cd $yearDir
        mkdir -p logFiles # Make logFiles directory if it doesn't exist
        echo "${YEL}Log directory: ${yearDir}/logFiles${NC}"
        newtxt="fail.txt"

        # declare -a massPnts=()
        for part in ${myParticles[*]}; do # Loop over particles
            # Currently waits 2 seconds between each 'crab submit'. This takes longer, but ensures no jobs get skipped.
            # Could run loop without waiting, but would need a check at the end of the script to ensure everything was submitted.
            # Waiting the extra time is simpler and safer for now. However, some experimental code for checking submissions has been left in this file.
            listOfScripts=config/crab*$part*.py
            echo "${YEL}Submitting crab jobs in ${GRN}$yearDir ${YEL}for ${PURP}$part${NC}"

            for job in $listOfScripts; do
                trimstring=${job#*"/"} # Trims the config/ from front of string
                crabName=${trimstring%"."*} # Trims .py from back of string
                crab submit $job >> logFiles/$crabName.txt 
                sleep 2s # If crab jobs are submitted too quickly, some don't go through
                # massPnts+=( ${crabName##"c"*"_"} ) # Trims the everything but the mass point/momentum

            done
        done
                
        echo "${YEL}Exiting $yearDir...${NC}"
        cd ..
    done
done

echo "jobs submitted"


#==================================================================================
# Check Submission ////////////////////////////////////////////////////////////////
#==================================================================================

# Currently, script submits jobs in series. This takes longer but eliminates risk of lost jobs.
# If the jobs are submitted in the background with '&', they can be rapidly submitted to CRAB in parallel. 
# However, this sometimes causes jobs to be 'lost', and not submit successfully.
# Below is some test code (DOES NOT WORK YET) that would handle this issue.

# for dat in ${myDatatypes[*]}; do # Loop over mc and data
#     if [[ $dat == "data" ]] ; then continue; fi #skips the loop for data, not implemented yet

#     for yr in ${myYears[*]}; do # Loop over years
#         # if [[ $yr != "2017" ]]; then continue; fi
#         yearDir="jobs$yr"
#         mkdir -p $yearDir # Make sure $yearDir exists
#         echo "${YEL}Entering $yearDir...${NC}"
#         cd $yearDir
#         mkdir -p logFiles # Make logFiles directory if it doesn't exist
#         echo "${YEL}Log directory: ${yearDir}/logFiles${NC}"
#         newtxt="fail.txt"

#         declare -a massPnts=()
        
#         echo "nap time"
#         sleep 60s
#         echo "checking dirs"
#         for m in ${massPnts[*]}; do
#             echo $m
#             bestDir="CrabBEST/"*"_${mass}_trees"*
#             if [[ ! -d $bestDir ]]; then
#                 echo "sad dir"
#                 echo "Error: $bestDir did not submit. Submitting..." >> $newtxt
#                 # crab submit $f >> logFiles/$crabName.txt &
#             fi
#         done
#     done
# done

# # Use this to kill everything that was just submitted--useful for testing script
# for d in */CrabBEST/*/ ; do
#     # echo $d | cut -d '/' -f 2 
#     crab kill -d $d
# done

# echo "jobs killed"

#Undo the alias used for this script
unalias echo
