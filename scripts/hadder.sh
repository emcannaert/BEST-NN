#!/bin/bash                                                                                                                                                            
#=========================================================================================                                                                             
# hadder.sh ------------------------------------------------------------------------------                                                                             
#-----------------------------------------------------------------------------------------                                                                             
# Author(s): Mark Samuel Abbott ----------------------------------------------------------                                                                             
#-----------------------------------------------------------------------------------------                                                                             
SECONDS=0 # Reset bash seconds timer
startdate=$(date) #Save start date
timelog="timelog_hadder.txt"

# Save color outputs
YEL='\033[93m' # Yellow
NC='\033[0m' # No Color

# By particle, by mass point, find BESTInputs files, hadd them together
# hadded file will be in local dir. Send file to eos, delete local file, repeat. mv to temp dir to do this
# eval `scramv1 runtime -sh` # This is the alias to cmsenv

declare -a allParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")
# declare -a allParticles=("HH")
eosDirPath="/store/user/maabbott/"
echo -e "${YEL}Will combine root files at $eosDirPath for ${allParticles[@]} ${NC}"

tempDir="temp_hadder"
mkdir -p $tempDir # Make sure $tempDir exists
echo -e "${YEL}Entering $tempDir ${NC}"
cd $tempDir
haddFile="BESTInputs_all.root"
if [ -f $haddFile ] ; then rm $haddFile; fi

# List all dirs in eos space
echo -e "${YEL}Checking directories at $eosDirPath ${NC}"
allDirs=`xrdfs root://cmseos.fnal.gov ls ${eosDirPath}`

# counter=0
for part in ${allParticles[*]}; do
    echo -e "${YEL}Beginning ${part} ${NC}"
    partDirs=`grep .*$part <<< "$allDirs"` # Select appropriate particles from full dir list

    for partDir in ${partDirs[*]}; do
        echo -e "${YEL}Combining root files for: ${partDir} ${NC}"
        eosBESTFiles=`xrdfs root://cmseos.fnal.gov ls -u -R ${partDir} | grep '.*211124*.*BEST'` # Find all BESTInputs*.root file paths for this dir
        # echo "BESTFiles are ${eosBESTFiles[*]}"
        # Loop through entries
        declare -a goodFiles=()
        for rootFile in ${eosBESTFiles[*]}; do
            if [[ $rootFile =~ $haddFile ]]; then eos root://cmseos.fnal.gov rm -rf "/store${rootFile#*"//store"}"; # If combined file exists already, remove it
            # if [[ $rootFile =~ $haddFile ]]; then continue 2; # If combined file exists already, keep it and skip this dir
            else                                  goodFiles+=( "$rootFile" );              fi
        done
        rootDir=${goodFiles[0]%"BESTInputs"*} # Grab one of the root file paths, and trim the 'BESTInputs_1.root' part off. Now $rootDir is the dir containing the root files, which we will copy our hadded root file to

        if [[ ${#goodFiles[@]} == 1 ]]; then
            xrdcp $goodFiles ${rootDir}${haddFile}
        else
            hadd $haddFile ${goodFiles[@]}
            xrdcp $haddFile ${rootDir}${haddFile}
            rm $haddFile
        fi

        # ((counter++))
        # if [[ $counter == 2 ]]; then break; fi

    done

done

echo -e "${YEL}Leaving and removing $tempDir ${NC}"
cd ..
rm -r $tempDir

echo -e "${YEL}Done hadding all files in${NC} ${eosDirPath}${YEL}!${NC}"
echo "Script began on $startdate, and ran for $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec" >> $timelog