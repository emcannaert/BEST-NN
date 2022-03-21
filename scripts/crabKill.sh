#!/bin/bash
#=========================================================================================
# crabKill.sh ----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script kills each crab job in each directory.

################################## NOTES TO SELF ##################################
# Check error flags?


echo "\nKilling jobs..."
logFile="logKill.txt"
pids=
# Kill crab jobs
for job in */CrabBEST/*/ ; do
    # echo $job | cut -d '/' -f 2 
    crab kill -d $job >> $logFile &
    pids+=" $!"
done

wait $pids # Wait until all crab status commands are done
echo "\nKilling complete. Check output at $logFile"