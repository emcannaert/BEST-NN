#!/bin/bash
#=========================================================================================
# deleteEOS.sh ---------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott, Johan S. Bonilla ----------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory.
# This script deletes every directory in the user's eos space.

################################## NOTES TO SELF ##################################
# Update this to not have to use dirs.txt. xrdfsls was failing because of the alias. 
# Use 'xrdfs root://cmseos.fnal.gov ls -u' instead, should work
# Include code to delete certain jobs? Certain years, particles, mass points, timestamps?

# eosDirPath="/store/user/msabbott/"
# echo "Listing files in $eosDirPath"
# eosDirs=`xrdfsls $eosDirPath`
# declare -a processes
# processes=("ToZZ" "ToTT" "QCD_Pt" "ToBB" "ToHH" "Tohh" "ToWW")
# for dir in $eosDirs; do
#     #echo "Try $dir"
#     for myProcess in "${processes[@]}"; do
# 	if grep -q .*"$myProcess".* <<< "$dir"; then
# 	    echo "Deleting $dir"
# 	    eosrm -rf $dir
# 	fi
#     done
# done

# The below code is a work around. Should edit code above and use that instead.

# do eosls /store/user/msabbott/ >> dirs.txt first
file="dirs.txt"
# Delete old eos samples
while read line; do # Reads the previous sample file, stores values to arrays
    fullPath="/eos/uscms/store/user/msabbott/$line"
    # fullPath="/eos/user/m/msabbott/$line"
    echo "deleting $fullPath"
    eos root://cmseos.fnal.gov rm -rf $fullPath
    # echo "deleting $line"
    # eos root://cmseos.fnal.gov rm -rf $line
done < "dirs.txt"
