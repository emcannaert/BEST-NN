#!/bin/bash
#=========================================================================================
# crabStatus.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Samantha Abbott -------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script checks the status of each crab job in each directory.

YEL='\033[93m' # Yellow
NC='\033[0m' # No Color

logFile="Logs/statusLog.txt"
echo >> $logFile
jobFile="Logs/jobsToCheck.txt"
echo >> $jobFile

declare -a jobsToCheck=()
declare -a unfinishedJobs=()
echo -e "\n${YEL}Checking jobs...${NC}"

pids= 
allJobs=0
finishedJobs=0

file=$jobFile

########## this block of code should be uncommented after the first run of crabStatus
########## double-commented out things can remain commented
# # OLDIFS=$IFS # Preserve the old IFS to reinstate it later
# # IFS='\n'
# while read -r job; do
#     # echo $job
#     jobsToCheck+=( "$job" )
# done < $jobFile
# # IFS=$OLDIFS # Resets $IFS so the rest of the code works
#####################
########### Below, switch the for loop after the first run of crabStatus

# Check job status of crab jobs:
for job in */CrabBEST/*/ ; do
# for job in ${jobsToCheck[*]}; do
    ((allJobs++))
    # echo -e "\n$job"
    # echo "test"
    # echo $job
    # output=$(crab status $job | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)')
    # output=$(/cvmfs/cms.cern.ch/common/crab status $job | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)')
    output=$(/cvmfs/cms.cern.ch/common/crab status $job)

    if [[ ("$output" =~ "SUBMITTED") && ("$output" =~ "finished") && ("$output" =~ "100.0") ]]; then
        ((finishedJobs++))
        continue
    else
        unfinishedJobs+=( "$job" )
        echo "$output" >> $logFile
        # echo "$output"
        echo "$output" | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)'
        echo -e '\n'
        # /cvmfs/cms.cern.ch/common/crab kill $job
    fi
    # echo "$output" >> $logFile

    if [[ ("$output" =~ "FAILED") && ("$output" =~ "failed") ]]; then
        /cvmfs/cms.cern.ch/common/crab resubmit $job
        echo -e '\n'
    fi
    
    # if [[ "$output" =~ "dagman" ]]; then
    #     /cvmfs/cms.cern.ch/common/crab kill $job
    #     rm -r $job
    #     # if [[ "$job" =~ "dagman" ]]; then
    #     echo -e "\nSUBMIT AGAIN $job\n "

    #     # /cvmfs/cms.cern.ch/common/crab submit $job
    # fi

    # Use this to record the most important info only: 
    # crab status $job | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)' >> $logFile 
    
    # Use this to record the entire output:
    # crab status $job >> $logFile 
    
    echo -e "----------------------\n\n----------------------" >> $logFile
    pids+=" $!"
    # echo "$job" >> $logFile 
    # output=`crab status $job | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)'`
    # if "finished     		100.0%" =~ 
done

# # echo ${pids[*]}
# wait $pids # Wait until all crab status commands are done
echo "###############################" >> $logFile

rm $jobFile
for job in ${unfinishedJobs[*]}; do
    echo "$job" >> $jobFile
done



# sort -o $logFile{,} # Sorting is difficult, since outputs come in randomly...
echo -e "\nFinished checking jobs." 
echo -e "${YEL}${finishedJobs}/${allJobs} jobs complete${NC}" 
echo -e "${YEL}Find unfinished jobs at${NC} $logFile" 