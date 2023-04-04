#!/bin/bash
#=========================================================================================
# crabKill.sh ----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Samantha Abbott -------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script kills each crab job in each directory.

################################## NOTES TO SELF ##################################
# Check error flags?


logFile="Logs/killLog.txt"
echo >> $logFile

echo "Killing jobs..."
pids=
# Kill crab jobs
for job in */CrabBEST/*/ ; do
    # echo $job | cut -d '/' -f 2 
    crab kill -d $job >> $logFile &
    pids+=" $!"
done

wait $pids # Wait until all crab status commands are done
echo -e "\nKilling complete. Check output at $logFile"