#!/bin/bash
#=========================================================================================
# crabStatus.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Sam Abbott -------- ---------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script checks the status of each crab job in each directory.

################################## NOTES TO SELF ##################################
# Needs to be updated to handle multiple years
# Edit crabStatus for new format, check number of submissions? Pipe output to text file? Curate a summary? Auto resubmit?
# Put in flags that do things?
#   grep status, check it, assign TRUE/FALSE flag to dictionary that contains crab file directory? Then resubmit or display just those or output to terminal
##### assign output to var. check for 'finished 100%'. if yes just print dir or nothing. if no print more info


YEL='\033[93m' # Yellow
NC='\033[0m' # No Color

logFile="Logs/statusLog.txt"
jobFile="Logs/jobsToCheck.txt"
declare -a jobsToCheck=()
declare -a unfinishedJobs=()
echo -e "\n${YEL}Checking jobs...${NC}"

pids= 
allJobs=0
finishedJobs=0

file=$jobFile
# OLDIFS=$IFS # Preserve the old IFS to reinstate it later
# IFS='\n'
while read -r job; do
    # echo $job
    jobsToCheck+=( "$job" )
done < $jobFile
# IFS=$OLDIFS # Resets $IFS so the rest of the code works
# exit
# echo "read"
# Check job status of crab jobs:
# for job in */CrabBEST/*/ ; do
for job in ${jobsToCheck[*]}; do
    ((allJobs++))
    echo $job
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
    fi
    if [[ "$output" =~ "FAILED" ]]; then
        /cvmfs/cms.cern.ch/common/crab resubmit $job
    fi


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
echo -e "\n${YEL}${finishedJobs}/${allJobs} jobs complete.${NC} $logFile" 
echo -e "\n${YEL}Find unfinished jobs at${NC} $logFile" 