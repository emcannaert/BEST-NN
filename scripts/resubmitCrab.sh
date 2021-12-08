#!/bin/bash

#106

# Resubmit crab jobs (currently resubmits all jobs, only the failed jobs will acutally resubmit)
echo -e "\n${YEL}Resubmitting jobs..."
logFile="logResubmit.txt"
pids=
for d in */CrabBEST/*/ ; do
    if [[ "$d" == *"QCD"* ]]; then continue; fi
    # echo $d | cut -d '/' -f 2 
    crab resubmit -d $d >> $logFile 
    # pids+=" $!"
done

# wait $pids
echo -e "\n${YEL}Resubmit complete. Check output at $logFile"
