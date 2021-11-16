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
#     crab status $d | grep 'Jobs status'
# done

YEL='\033[93m' # Yellow
NC='\033[0m' # No Color

# # Re-spawn as a background process, if we haven't already.
# if [[ "$1" != "-n" ]]; then
#     nohup "$0" -n &
#     exit $?
# fi

# if [[ "$1" != "-n" ]]; then
#     $0 -n & disown
#     exit $?
# fi

# Put in flags that do things?

checkFile="logCheck.txt"
echo -e "\n${YEL}Checking jobs...${NC}"
pids=
for d in */CrabBEST/*/ ; do
    # echo $d | cut -d '/' -f 2 
    # echo "$d" >> $checkFile
    crab status $d | grep -E '(CRAB project directory|Status on the CRAB server|Jobs status)' >> $checkFile 
    pids+=" $!"
    # crab status $d >> $checkFile &
done
# echo ${pids[*]}
wait $pids 
echo "###############################" >> $checkFile
# sort -o $checkFile{,}
echo -e "\n${YEL}Finished checking jobs. Find complete output at${NC} $checkFile" 
