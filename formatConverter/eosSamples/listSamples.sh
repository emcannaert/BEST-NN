#!/bin/bash
#=========================================================================================
# listSamples.sh -------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Johan S Bonilla, Brendan Regnery, Sam Abbott --------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/formatConverter/eosSamples directory.
# This script lists each BESTInputs file on eos, for each particle, for each year.

################################## NOTES TO SELF ##################################
# Edit this to match style of other scripts

# List files from eos with BEST in name (typically /eos/path/BESTInputs_*.root)
eosDirPath="/store/user/maabbott/"
echo "Listing files in $eosDirPath"
if [ $# -gt 0 ]; then
    echo "Your command line contains $# arguments."
else
    echo "Your command line contains no arguments, please specify what years and processes you'd like to submit."
    echo "Options: all, 2016_APV,  2016, 2017, 2018, HH, BB, TT, WW, ZZ, QCD"
    echo "If you specify all, all years and samples are made. Else the outer product of the arguments are done, e.g. '2015 2016 HH TT BB' will make 6 files."
    echo "Example: ./listHiggsSamples.sh 2017 2018 HH TT BB"
    echo "Example: ./listHiggsSamples.sh all"
    exit 1
fi
declare -a myYears
declare -a processes
unset myYears
unset processes
if [ $1 == "all" ]; then
    echo "Making list for all years and samples"
    # myYears=("2016_APV" "2016" "2017" "2018")
    # myYears="2017"
    myYears="2016"
    processes=("BB" "HH" "TT" "WW" "ZZ" "QCD" ) 
    # processes=("RSG" ) 
    # processes=("HH" ) 
else
    for arg in "$@"; do
        if [ $arg == "2016_APV" ] || [ $arg == "2016" ] || [ $arg == "2017" ] || [ $arg == "2018" ]; then
            echo "Adding $arg to years"
            myYears+=($arg)
        else
            echo "Adding $arg to processes"
            processes+=($arg)
        fi
    done
fi
echo "Making sample lists for ${processes[@]} in years ${myYears[@]}"
# eosBESTFiles=`xrdfsls -R $eosDirPath`
eosBESTFiles=`xrdfs root://cmseos.fnal.gov ls -R $eosDirPath`
# echo $eosBESTFiles
for year in "${myYears[@]}"; do
    for process in "${processes[@]}"; do
        echo "Making list for $year $process"
        # eosBESTFiles=`xrdfsls -R $eosDirPath | grep '.*HH.*211124*.*BEST'`

        filesToAdd=`grep .*$process.*$year.*BEST <<< "$eosBESTFiles"`
        # declare -a tempfiles
        # for f in $filesToAdd; do
        #     if [[ "$year" == "2016" ]] && [[ "$f" =~ "2016_APV" ]]; then continue; fi # year = 2016 will match all the 2016_APV files, so skip them

        #     tempstring=(${f%"/0000"*})   
        #     if [[ ${tempfiles[*]} =~ $tempstring ]]; then continue; fi
        #     echo $tempstring
        #     tempfiles+=($tempstring)   
        # done
        # echo -e "\n"

        # filesToAdd=`grep .*$process.*BEST$year.*211124*.*BESTInputs.*.root <<< "$eosBESTFiles"`
        # filesToAdd=`grep .*$process.*BEST <<< "$eosBESTFiles"`
        
        # Check if file exists, if so delete
        # fileToWrite="listOf$process""FilePaths$year.txt"
        
        if [[ "$process" == "TT" ]]; then # Create an extra dataset for the TTbar with just RSG samples
            fileToWrite="listOfRSGFilePaths${year}.txt"
            if [ -f $fileToWrite ] ; then rm $fileToWrite; fi
            # Write each BESTInput-file's xrootd path to fileToWrite
            for f in $filesToAdd; do
                if [[ "$year" == "2016" ]] && [[ "$f" =~ "2016_APV" ]]; then continue; fi # year = 2016 will match all the 2016_APV files, so skip them
                if [[ ! "$f" =~ "RSGluon" ]]; then continue; fi # only include RSGluon files in this set
                echo "root://cmsxrootd.fnal.gov/$f" >> $fileToWrite
            done      
            echo "Checkout your new list of files at $fileToWrite"        
        fi
        fileToWrite="listOf$process""FilePaths$year.txt"

        if [ -f $fileToWrite ] ; then rm $fileToWrite; fi
        # Write each BESTInput-file's xrootd path to fileToWrite
        for f in $filesToAdd; do
            if [[ "$year" == "2016" ]] && [[ "$f" =~ "2016_APV" ]]; then continue; fi # year = 2016 will match all the 2016_APV files, so skip them
            if [[ "$process" == "TT" ]] && [[ "$f" =~ "RSGluon" ]]; then continue; fi # do not include RSGluon files in this set

            echo "root://cmsxrootd.fnal.gov/$f" >> $fileToWrite
        done
        
        echo "Checkout your new list of files at $fileToWrite"
    done
done
