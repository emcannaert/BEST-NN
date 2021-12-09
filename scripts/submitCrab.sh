#!/bin/bash
#=========================================================================================
# submitCrab.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------
#135

# Better PATH variables? 
# Check that all the jobs submit (is 5 seconds good enough?)
# Edit submitCrab/creatConfig to display number of files per dir for samples + submissions (prolly bash); grep lines of sample file, grep files in dirs
# could prolly print the output by reading in the sample file arrays with the fillscripts code and compare that to a grep of the CrabBEST dir, automatically submit
# Edit checkcrab for new format, check number of submissions? Pipe output to text file? Curate a summary? Auto resubmit?
# Edit kill jobs?
# Can I run crab in parallel?
# Test that specific cases run correctly
# Make log file, config file, and crab dir uniform?


# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script takes options/arguments from the user and submits the appropriate jobs to crab. 
# This script will also call createConfig.py to generate the crab config files to submit.
# The script lives in the scripts directory and the symbolic links in each of the submit201X directories should be executed within their respective directories.


################# These lines of code that parse in the arguments are nearly identical to fillSamples.sh:

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

# Check that user provided valid amount of options, exit if not. The only valid inputs would have 0 or 6 options/arguments.
if [[ $# != 0 ]] && [[ $# != 6 ]]; then 
    echo "${YEL}WOAH, slow down there friend!${NC} Your command line contains $# options/arguments!"
    echo "Please pass options and arguments as:"
    echo
    echo "./submitCrab ${CYAN}-p \"<particle 1> <particle 2>...\" ${GRN}-y \"<year 1> <year 2>...\" ${PURP}-d <datatype>${NC}"
    echo
    echo "${CYAN}Particle arguments: ${BLUE}all${NC}, ${YEL}or${NC} any combination of ${CYAN}QCD, HH, WW, ZZ, tt, bb${NC}"
    echo "${GRN}Year arguments: ${BLUE}all${NC}, ${YEL}or${NC} any combination of ${GRN}2016_APV, 2016, 2017, 2018${NC}"
    echo "${PURP}Datatype arguments: ${BLUE}all${NC} ${YEL}or${NC} ${PURP}mc${NC} ${YEL}or${NC} ${PURP}data${NC}"
    echo "${YEL}Note that for the ${GRN}year${YEL} arguments, ${GRN}2016_APV${YEL} is a special case. It corresponds to the 2015 datasets, but in DAS it us under 2016 with APV in the dataset name."
    echo
    echo "All options and arguments are case-sensitive, and all options-argument pairs can be executed in any order."
    echo "${YEL}Example:${NC} ./submitCrab.sh ${PURP}-d mc ${GRN}-y ${BLUE}all ${CYAN}-p HH${NC}"
    echo
    echo "Quotes are necessary when passing multiple arguments for one option."
    echo "${YEL}Example:${NC} ./submitCrab.sh ${GRN}-y \"2016_APV 2016 2017\" ${PURP}-d data ${CYAN}-p \"HH WW\"${NC}"    
    echo
    echo "Simply running the script without any options or arguments selects the '${BLUE}all${NC}' argument for ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype.${NC}"
    echo "${YEL}Example:${NC} ./submitCrab.sh"
    exit 1
fi

# Declare the full list of valid arguments for each option
declare -a allParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")
declare -a allYears=("2016_APV" "2016" "2017" "2018")
declare -a allDatatypes=("mc" "data")
# Declare initial arrays to fill with user chosen arguments later
declare -a myParticles
declare -a myYears
declare -a myDatatypes

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
                            echo "${YEL}Run script without any options to see usage:${NC} ./submitCrab.sh"
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
                            echo "${YEL}Run script without any options to see usage:${NC} ./submitCrab.sh"
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
                            echo "${YEL}Run script without any options to see usage:${NC} ./submitCrab.sh"
                            echo "${RED}Exiting without creating samples...${NC}"
                            exit 1
                        fi
                    done
                fi
            ;;
            \?) # Catches invalid options
                echo "${YEL}Error:${NC} Invalid option: -$OPTARG"
                echo "${YEL}Run script without any options to see usage:${NC} ./submitCrab.sh"
                echo "${RED}Exiting without creating samples...${NC}"
                exit 1
            ;;
            :)  # Catches options missing arguments
                echo "${YEL}Error:${NC} Option -$OPTARG requires an argument."
                echo "${YEL}Run script without any options to see usage:${NC} ./submitCrab.sh"
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
if [[ $(voms-proxy-info -timeleft) > 3600 ]] && [[ $(voms-proxy-info -vo) == "cms" ]]; then
    echo "${GRN}Valid proxy confirmed!${NC}"
    echo
else
    echo "${YEL}Error: Proxy either doesn't exist or will expire soon. Initializing new proxy...${NC}"
    voms-proxy-init --valid 192:00 -voms cms
    echo
fi

################# At this point, the code unique to this file begins: 

# Call createConfig.py to generate the config files:
python createConfig.py -p ${myParticles[*]} -y ${myYears[*]} -d ${myDatatypes[*]}


eval `scramv1 runtime -sh` # This is the alias to cmsenv
source /cvmfs/cms.cern.ch/crab3/crab.sh # Source crab

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

        declare -a massPnts=()
        for part in ${myParticles[*]}; do # Loop over particles

            # UPDATE THIS TO USE ARRAYS INSTEAD OF WHATS IN THE DIR? COULD BE A WAY TO AUTOCHECK IF EVERYTHING SUBMITS
            # wait this should just use the list of crab dirs that I already made...

            #so copy code to read in file, then store prolly just the crab dir, and iterate over that to check everything.
            # to submit everything, just recreate the python code that creates the config file; the mass point is right there. iterate over those to submit everything
            # compare the two to find if something is a sample but didnt get submitted
            # automatically submit that job, maybe by adding the relevant array values to a separate list and having a loop of code iterate over that through an if statement
            # could maybe reuse this code for crab resubmit/crab check (should be the same, maybe an option to NOT resubmit)
            # To test this, make sure crabkill works so crab jobs can be cancelled
            # ask johan: parallel crab? how many scripts (better many scripts that do specific things, even if overlap in code, or fewer scripts that you give options to??)

            # Maybe try not running it in background
            
            listOfScripts=config/crab*$part*.py
            echo "${YEL}Submitting crab jobs in ${GRN}$yearDir ${YEL}for ${PURP}$part${NC}"

            for f in $listOfScripts; do
                trimstring=${f#*"/"} # Trims the config/ from front of string
                crabName=${trimstring%"."*} # Trims .py from back of string
                crab submit $f >> logFiles/$crabName.txt &
                # sleep 1s # If crab jobs are submitted too quickly, some don't go through
                massPnts+=( ${crabName##"c"*"_"} ) # Trims the everything but the mass point/momentum

            done
            # Could rapid submit all files, add masses to list, then loop over them to check
        done
        
        echo "nap time"
        sleep 60s
        echo "checking dirs"
        for m in ${massPnts[*]}; do
            echo $m
            bestDir="CrabBEST/"*"_${mass}_trees"*
            if [[ ! -d $bestDir ]]; then
                echo "sad dir"
                echo "Error: $bestDir did not submit. Submitting..." >> $newtxt
                # crab submit $f >> logFiles/$crabName.txt &
            fi
        done


        
        echo "${YEL}Exiting $yearDir...${NC}"
        cd ..
        # echo "${YEL}Checkout your new list of files at:${NC} $fileToWrite"
    done
done

echo "jobs submitted"
sleep 60s

# auto kill
for d in */CrabBEST/*/ ; do
    # echo $d | cut -d '/' -f 2 
    crab kill -d $d
done

echo "jobs killed"

#Undo the alias used for this script
unalias echo