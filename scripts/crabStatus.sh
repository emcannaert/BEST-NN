#!/bin/bash
#=========================================================================================
# crabStatus.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott, Johan S Bonilla -----------------------------------------
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

logFile="logStatus.txt"
echo -e "\n${YEL}Checking jobs...${NC}"
pids= 
# Check job status of crab jobs.
for job in */CrabBEST/*/ ; do
    # echo "$job" >> $logFile 
    crab status $job | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)' >> $logFile 
    # output=`crab status $job | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)'`

    # if "finished     		100.0%" =~ 
    # crab status $job >> $logFile # Use this to pipe entire output to file
    echo -e "----------------------\n\n----------------------" >> $logFile
    pids+=" $!"
done
# echo ${pids[*]}
wait $pids # Wait until all crab status commands are done
echo "###############################" >> $logFile
# sort -o $logFile{,} # Sorting is difficult, since outputs come in randomly...
echo -e "\n${YEL}Finished checking jobs. Find complete output at${NC} $logFile" 