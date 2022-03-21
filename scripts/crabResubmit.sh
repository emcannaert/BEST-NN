#!/bin/bash
#=========================================================================================
# crabResubmit.sh ------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script resubmits each crab job in each directory.

################################## NOTES TO SELF ##################################
# Resubmit crab jobs (currently resubmits all jobs, only the failed jobs will acutally resubmit)
# Consider connecting to crabStatus.sh, and resubmit jobs that have failed?

echo "Resubmitting jobs..."
logFile="logResubmit.txt"
pids=
for job in */CrabBEST/*/ ; do
    # echo $job | cut -d '/' -f 2 
    crab resubmit -d $job >> $logFile 
    # pids+=" $!"
done

# wait $pids
echo "\nResubmit complete. Check output at $logFile"
