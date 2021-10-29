#!/bin/bash                                                                                                                                                            
#=========================================================================================                                                                             
# delete.sh --------------------------------------------------------------------                                                                             
#-----------------------------------------------------------------------------------------                                                                             
# Author(s): Johan S Bonilla -------------------------------------------------------------                                                                             
#-----------------------------------------------------------------------------------------                                                                             

# Delete old eos samples

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

# do eosls /store/user/msabbott/ >> dirs.txt first
file="dirs.txt"

while read line; do # Reads the previous sample file, stores values to arrays
    fullPath="/eos/uscms/store/user/msabbott/$line"
    # fullPath="/eos/user/m/msabbott/$line"
    echo "deleting $fullPath"
    eos root://cmseos.fnal.gov rm -rf $fullPath
    # echo "deleting $line"
    # eos root://cmseos.fnal.gov rm -rf $line
done < "dirs.txt"
