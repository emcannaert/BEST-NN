#!/bin/bash
#=========================================================================================
# checkJobs.sh ----------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Johan S Bonilla -------------------------------------------------------------
#-----------------------------------------------------------------------------------------

# Check job status of crab jobs.
# grep the output of crab status to terminal and see who isn't 100% done
# for d in CrabBEST/*/ ; do
#     echo $d | cut -d '/' -f 2 
#     crab kill -d $d
# done

# Allow option parsing?
# Check job status of crab jobs.
# grep the output of crab status to terminal and see who isn't 100% done?
# Error flags?
echo -e "\n${YEL}Killing jobs..."
killFile="logKill.txt"
pids=
for d in */CrabBEST/*QCD*/ ; do
    # echo $d | cut -d '/' -f 2 
    crab kill -d $d >> $killFile &
    pids+=" $!"
done

wait $pids
echo -e "\n${YEL}Killing complete. Check output at $killFile"
