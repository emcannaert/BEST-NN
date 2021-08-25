#!/bin/bash
#=========================================================================================
# fillSamples.sh --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott -------------------------------------------------------------
#-----------------------------------------------------------------------------------------

#This script lives in the scripts directory and fills text files in the samples directory. Moving it will cause problems with '$fileToWrite', but you can modify the path.
#This script takes arguments for particle, year, and datatype to create/fill the sample files with dataset names, using dasgoclient.

######################################### NOTES TO SELF ############################
# Need something remind the user to do voms
# Add exception in dasCheck just in case it returns two values
# Things to look into:
# HH: 60000 mass point instead of 6000 on DAS for HH for all years (lol), DAS is missing mass points: (2017: 2000), (2018: 6500)
# WW: DAS is missing mass point: (2015: 5000)
# ZZ: DAS is missing mass points: (2015: 600, 1600), (2017: 2000, 3000, 3500, 5000), (2018: 1000, 1400, 1800)
# bb: DAS is missing mass point: (2018: 500) 5 6 8 10
# tt: Mass points on DAS not scanned for certain years: (2015 and 2016: 400, 700, 900)
# tt: DAS is missing mass points: (all years: 5000, 5500, 6000, 6500, 7000, 7500, 8000), (2017 and 2018: 500, 600, 800, 1000)
# Johan indicated that for the tt samples, there would be multiple samples at each mass point, but with different widths.
# I only see one width per mass point...but this code is easily adaptable in case there are more samples with different widths.


#Define ANSI colors here for the output since I am extra:
RED='\033[0;31m' # Red
CYAN='\033[1;36m' # Light Cyan
BLUE='\033[0;34m' # Blue
PURP='\033[1;35m' # Light Purple
GRN='\033[1;32m' # Light Green
YEL='\033[1;33m' # Yellow
BRN='\033[0;33m' # Brown/Orange
NC='\033[0m' # No Color
#This alias makes the script simpler, as '-e' is needed to print color. This is undone by 'unalias' at the end of the code.
shopt -s expand_aliases
alias echo='echo -e'


if [[ $# != 1 ]] && [[ $# != 6 ]]; then # Check that user provided options, exit if not. The only valid inputs would have 1 or 6 options/arguments.
    echo "${YEL}WOAH, slow down there friend!${NC} Your command line contains $# options/arguments!"
    echo "Please pass options and arguments as:"
    echo
    echo "${CYAN}-p \"<particle 1> <particle 2>...\" ${GRN}-y \"<year 1> <year 2>...\" ${PURP}-d <datatype>${NC}"
    echo
    echo "${CYAN}Particle arguments: ${RED}all${NC}, ${YEL}or${NC} any combination of ${CYAN}QCD, HH, WW, ZZ, tt, bb${NC}"
    echo "${GRN}Year arguments: ${RED}all${NC}, ${YEL}or${NC} any combination of ${GRN}2015, 2016, 2017, 2018${NC}"
    echo "${PURP}Datatype arguments: ${RED}all${NC} ${YEL}or${NC} ${PURP}mc${NC} ${YEL}or${NC} ${PURP}data${NC}"
    echo
    echo "All options and arguments are case-sensitive, and all options-argument pairs can be executed in any order."
    echo "${YEL}Example:${NC} ./fillSamples.sh ${PURP}-d mc ${GRN}-y ${RED}all ${CYAN}-p HH${NC}"
    echo
    echo "Quotes are necessary when passing multiple arguments for one option."
    echo "${YEL}Example:${NC} ./fillSamples.sh ${GRN}-y \"2015 2016 2017\" ${PURP}-d data ${CYAN}-p \"HH WW\"${NC}"    
    echo
    echo "Simply passing '${RED}-a${NC}' as the only option selects the '${RED}all${NC}' argument for ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype.${NC}"
    echo "${YEL}Example:${NC} ./fillSamples.sh ${RED}-a${NC}"
    exit 1
fi

# Declare the full list of valid argument for each option
declare -a allParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")
declare -a allYears=("2015" "2016" "2017" "2018")
declare -a allDatatypes=("mc" "data")
# Declare initial arrays to fill with user chosen arguments later
declare -a myParticles
declare -a myYears
declare -a myDatatypes

#This is where the options and arguments are parsed in.
while getopts :ap:y:d: opt; do
  case $opt in
    a) # The -a option. Chooses all samples.
        if [[ $# == 1 ]]; then #Checks for correct usage, exits if not.
            echo "\"${RED}all${NC}\" option triggered. ${RED}All${NC} samples for each ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype${NC}, will be generated."
            myParticles=${allParticles[*]}
            myYears=${allYears[*]}
            myDatatypes=${allDatatypes[*]}
        else
            echo "${YEL}Error:${NC} Invalid input. To run ${RED}all${NC} samples for each ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype${NC}, execute ./fillSamples.sh ${RED}-a${NC}"
            echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
            echo "${RED}Exiting without creating samples...${NC}"
            exit 1
        fi
        ;;      
    p)  #The -p option, for particles. Either fills "all" or the specificly chosen arguments.
        if [[ $OPTARG == "all" ]]; then
            myParticles=${allParticles[*]}
        else 
            for part in $OPTARG; do    
                if [[ ${allParticles[*]} =~ $part ]]; then #Check for valid arguments, then fills array.
                    myParticles+=($part)
                else #Invalid arguments trigger error message
                    echo "${YEL}Error:${NC} Invalid argument for ${CYAN}$opt${NC}: $part"
                    echo "Please choose ${RED}'all'${NC}, or the case-sensitive arguments: ${CYAN}${allParticles[*]}${NC}"
                    echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
                    echo "${RED}Exiting without creating samples...${NC}"
                    exit 1
                fi
            done
        fi
        ;;
    y)  #The -y option, for years. Either fills "all" or the specificly chosen arguments.
        if [[ $OPTARG == "all" ]]; then
            myYears=${allYears[*]}
        else
            for yr in $OPTARG; do
                if [[ ${allYears[*]} =~ $yr ]]; then #Check for valid arguments, then fills array.
                    myYears+=($yr)
                else #Invalid arguments trigger error message
                    echo "${YEL}Error:${NC} Invalid argument for ${GRN}$opt${NC}: $yr"
                    echo "Please choose ${RED}'all'${NC}, or the case-sensitive arguments: ${GRN}${allYears[*]}${NC}"
                    echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
                    echo "${RED}Exiting without creating samples...${NC}"
                    exit 1
                fi
            done
        fi
        ;;
    d)  #The -d option, for datatype. Either fills "all" or the specificly chosen arguments.
        if [[ $OPTARG == "all" ]]; then
            myDatatypes=${allDatatypes[*]}
        else
            for dat in $OPTARG; do
                if [[ ${allDatatypes[*]} =~ $dat ]]; then #Check for valid arguments, then fills array.
                    myDatatypes+=($dat)
                else ##Invalid arguments trigger error message
                    echo "${YEL}Error:${NC} Invalid argument for ${PURP}$opt${NC}: $dat"
                    echo "Please choose ${RED}'all'${NC}, or the case-sensitive arguments: ${PURP}${allDatatypes[*]}${NC}"
                    echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
                    echo "${RED}Exiting without creating samples...${NC}"
                    exit 1
                fi
            done
        fi
        ;;
    \?) # Catches invalid options
        echo "${YEL}Error:${NC} Invalid option: -$OPTARG"
        echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
        echo "${RED}Exiting without creating samples...${NC}"
        exit 1
        ;;
    :)  # Catches options missing arguments
        echo "${YEL}Error:${NC} Option -$OPTARG requires an argument."
        echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
        echo "${RED}Exiting without creating samples...${NC}"
        exit 1
        ;;
  esac
done

echo "${CYAN}Particle(s)${NC} selected: ${CYAN}${myParticles[*]}${NC}"
echo "${GRN}Year(s)${NC} selected: ${GRN}${myYears[*]}${NC}"
echo "${PURP}Datatype(s)${NC} selected: ${PURP}${myDatatypes[*]}${NC}"


#Define an array of mass points that were used to submit the samples. This will help in dicriminating between the orginal and v2 samples.
declare -a massPoints=("500" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000" "2500" "3000" "3500" "4000" "4500" "5000" "5500" "6000" "6500" "7000" "7500" "8000")

#Here we define a function to call dasgoclient to search the DAS, and assigns that string to "dasDataset". Does not do well if dasgoclient returns multiple outputs, so use carefully.
#This also defines two helpful strings, "dasYear" and "dasPart". 
checkDAS(){ ################### Takes inputs as: "checkDAS part year dat mass version" #########################
    #To search for our files on DAS, we need to manipulate the input data a bit.
    #Here we define "dasYear", which is used to search DAS.
    if [[ $2 == "2015" ]]; then #The Summer 2020 Ultra Legacy samples for 2015 are named "16MiniAODAPV"; the regular 2016 files do not have the "APV".
        dasYear="RunIISummer20UL16MiniAODAPV$5"
    else #All other years are straightforward. This trims the first two characters off of the string, so 2017 becomes 17, etc.
        dasYear="RunIISummer20UL${2:2}MiniAOD$5"
    fi
    #Here we build the case structure for each particle, and define "dasPart", which is used to search DAS.
    case $1 in
        "WW") # W case
            dasPart="BulkGravToWWToWhadWhad"
            dasDataset=( $(dasgoclient -query="dataset dataset=/${dasPart}_narrow_M-$4_T*/${dasYear}-*/MINIAODSIM datatype=$3") )
        ;;        
        "ZZ") # Z case
            dasPart="BulkGravToZZToZhadZhad"
            dasDataset=( $(dasgoclient -query="dataset dataset=/${dasPart}_narrow_M-$4_T*/${dasYear}-*/MINIAODSIM datatype=$3") )
        ;;        
        "bb") # b case
            dasPart="ZprimeToBB"
            dasDataset=( $(dasgoclient -query="dataset dataset=/${dasPart}_narrow_M-$4_T*/${dasYear}-*/MINIAODSIM datatype=$3") )
        ;;
        "HH") # Higgs case
            dasPart="GluGluToBulkGravitonToHHTo4B"
            dasDataset=( $(dasgoclient -query="dataset dataset=/${dasPart}_M-$4_narrow_T*/${dasYear}-*/MINIAODSIM datatype=$3") )
        ;;
        "tt") # Top case
            dasPart="ZprimeToTT"
            dasDataset=( $(dasgoclient -query="dataset dataset=/${dasPart}_M$4_W*_T*/${dasYear}-*/MINIAODSIM datatype=$3") )
        ;;
        "QCD") # QCD case
            echo "QCD is not implemented yet."
        ;;
    esac
    # echo $dasDataset
    #Note for future exception: Define dasDataset as an array, and add an exception that checks if dasDataset has more than one entry, then convert back? 
}

#This loop will create the text files, search for the datasets on DAS, check for version, and fill the text files accordingly.
for dat in ${myDatatypes[*]}; do #Loop over mc and data
    if [[ $dat == "data" ]] ; then continue; fi #skips the loop for data, not implemented yet
    echo "${PURP}Beginning $dat${NC}..."
    for yr in ${myYears[*]}; do #Loop over years
        echo "${GRN}Beginning $yr${NC}..."
        # Check if file exists, if so delete 
        fileToWrite="../samples/${dat}_${yr}.txt"
        echo "Writing samples to $fileToWrite"
        if [ -f $fileToWrite ] ; then rm $fileToWrite; fi
        for part in ${myParticles[*]}; do #Loop over particles
            if [[ $part == "QCD" ]] ; then continue; fi #skips the loop for QCD, not implemented yet
            echo "${CYAN}Beginning $part${NC}..."
            #These help keep track of how many datasets still need a version 2.
            v2counter=0
            v1counter=0        
            for mass in ${massPoints[*]}; do #Loop over mass points. Now we are checking each DAS dataset individually.
                checkDAS $part $yr $dat $mass "v2" #check DAS for a version 2 for this dataset
                if [[ ! -z "$dasDataset" ]]; then #This returns true if DAS found a version 2 dataset.
                    echo $dasDataset >> $fileToWrite #Write the dataset to the samples file
                    ((v2counter++)) #This increments v2counter
                else #If there is not a v2 dataset, check if there is a v1 dataset
                    checkDAS $part $yr $dat $mass #Leave version blank for checkDAS for version 1
                    if [[ ! -z "$dasDataset" ]]; then #This returns true if DAS found a version 1 dataset.
                        echo $dasDataset >> $fileToWrite #Write the dataset to the samples file                            
                        ((v1counter++)) #This increments v1counter
                    else #This returns a warning message that no dataset exists for these arguments.
                        # echo $dasDataset                        
                        echo "${YEL}No dataset exists${NC} for /${CYAN}${dasPart}${NC}*${mass}*/${GRN}${dasYear}${NC}*/MINIAODSIM"
                    fi
                fi
            done
            echo "For the /${CYAN}${dasPart}${NC}*/${GRN}${dasYear}${NC}*/MINIAODSIM datasets, there are ${v2counter} v2 datasets and ${v1counter} v1 datasets."
        done
        echo "Checkout your new list of files at: $fileToWrite"
    done
done
#Undo the alias used for this script
unalias echo