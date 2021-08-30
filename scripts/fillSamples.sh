#!/bin/bash
#=========================================================================================
# fillSamples.sh -------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, and fills text files in the BEST/samples directory. 
# The symbolic link in the BEST/preprocess directory should be executed when using this script.
# This script takes arguments for particle, year, and datatype to create/fill the sample files with dataset names from DAS, using dasgoclient.
# This script is specific to calling datasets for the Summer 2020 Ultra Legacy samples submitted by the UCD BEST team, searching for VLQs. But it can be modified to search for other datasets! 
# This script also checks for and keeps track of a version two, or "v2", for each dataset, as these updated datasets are still being produced as of writing this code.

######################################### NOTES TO SELF ############################
# Implement data
# Implement a quiet option?
# Change the file deleting code to simply delete the datasets that are being replaced, and not the entire file?
# HH: 60000 mass point instead of 6000 on DAS for HH for all years (checked this, the mass point is correctly 6000, the name is just wrong), DAS is missing mass points: (2017: 2000), (2018: 6500)
# WW: DAS is missing mass point: (2015: 5000)
# ZZ: DAS is missing mass points: (2015: 600, 1600), (2017: 2000, 3000, 3500, 5000), (2018: 1000, 1400, 1800)
# bb: DAS is missing mass point: (2018: 500)
# tt: Mass points on DAS not in the 21 mass points given: (2015 and 2016: 400, 700, 900) <- The script finds all samples, regardless of relation to the 21 mass points
# tt: DAS is missing mass points: (all years: 5000, 5500, 6000, 6500, 7000, 7500, 8000), (2017 and 2018: 500, 600, 800, 1000)
# tt: There should be multiple widths per mass point, but for now there is only one width per mass point. This code will need to be updated to handle widths correctly in the future.

# Define ANSI colors here for the output since I am extra:
RED='\033[91m' # Red
CYAN='\033[96m' # Light Cyan
BLUE='\033[94m' # Blue
PURP='\033[35m' # Light Purple
GRN='\033[92m' # Light Green
YEL='\033[93m' # Yellow
NC='\033[0m' # No Color
# BOLD = '\033[1m'
# UNDERLINE = '\033[4m'
#implement bold/underline?

#This alias makes the script simpler, as '-e' is needed to print color. This is undone by 'unalias' at the end of the code.
shopt -s expand_aliases
alias echo='echo -e'

# Check that user provided options, exit if not. The only valid inputs would have 1 or 6 options/arguments.
if [[ $# != 1 ]] && [[ $# != 6 ]]; then 
    echo "${YEL}WOAH, slow down there friend!${NC} Your command line contains $# options/arguments!"
    echo "Please pass options and arguments as:"
    echo
    echo "${CYAN}-p \"<particle 1> <particle 2>...\" ${GRN}-y \"<year 1> <year 2>...\" ${PURP}-d <datatype>${NC}"
    echo
    echo "${CYAN}Particle arguments: ${BLUE}all${NC}, ${YEL}or${NC} any combination of ${CYAN}QCD, HH, WW, ZZ, tt, bb${NC}"
    echo "${GRN}Year arguments: ${BLUE}all${NC}, ${YEL}or${NC} any combination of ${GRN}2015, 2016, 2017, 2018${NC}"
    echo "${PURP}Datatype arguments: ${BLUE}all${NC} ${YEL}or${NC} ${PURP}mc${NC} ${YEL}or${NC} ${PURP}data${NC}"
    echo
    echo "All options and arguments are case-sensitive, and all options-argument pairs can be executed in any order."
    echo "${YEL}Example:${NC} ./fillSamples.sh ${PURP}-d mc ${GRN}-y ${BLUE}all ${CYAN}-p HH${NC}"
    echo
    echo "Quotes are necessary when passing multiple arguments for one option."
    echo "${YEL}Example:${NC} ./fillSamples.sh ${GRN}-y \"2015 2016 2017\" ${PURP}-d data ${CYAN}-p \"HH WW\"${NC}"    
    echo
    echo "Simply passing '${BLUE}-a${NC}' as the only option selects the '${BLUE}all${NC}' argument for ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype.${NC}"
    echo "${YEL}Example:${NC} ./fillSamples.sh ${BLUE}-a${NC}"
    exit 1
fi

# Declare the full list of valid arguments for each option
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
            echo "\"${BLUE}all${NC}\" option triggered. ${BLUE}All${NC} samples for each ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype${NC}, will be generated."
            myParticles=${allParticles[*]}
            myYears=${allYears[*]}
            myDatatypes=${allDatatypes[*]}
        else
            echo "${YEL}Error:${NC} Invalid input. To run ${BLUE}all${NC} samples for each ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype${NC}, execute ./fillSamples.sh ${BLUE}-a${NC}"
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
                    echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${CYAN}${allParticles[*]}${NC}"
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
                    echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${GRN}${allYears[*]}${NC}"
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
                    echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${PURP}${allDatatypes[*]}${NC}"
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
echo
echo "${BLUE}Initiating DAS search. Checking voms cms proxy...${NC}"

# This checks for a voms cms proxy that will last longer than 30 minutes, and has the user create a new one if not
if [[ $(voms-proxy-info -timeleft) > 1800 ]] && [[ $(voms-proxy-info -vo) == "cms" ]]; then
    echo "${GRN}Valid proxy confirmed!${NC}"
    echo
else
    echo "${YEL}Error: Proxy either doesn't exist or will expire soon.${NC} Initializing new proxy..."
    voms-proxy-init --valid 192:00 -voms cms
    echo
fi

# These 21 mass points were originally used when submitting the GridPacks, but the DAS datasets don't exactly match them, so this script circumvents having to use these. 
# declare -a massPoints=("500" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000" "2500" "3000" "3500" "4000" "4500" "5000" "5500" "6000" "6500" "7000" "7500" "8000")

# Here we define a function to call dasgoclient to search the DAS, and assigns that string to "dasDatasets". Does not do well if dasgoclient returns multiple outputs, so use carefully.
# This also defines two helpful strings, "dasYear" and "dasPart". 
checkDAS(){ ################### Takes inputs as: "checkDAS particle year datatype" #########################

    # To search for our files on DAS, we need to manipulate the input data a bit. This is specific to our current analysis but can be modified for other analyses.
    # Here we define "dasYear", which is used to search DAS.
    if [[ $2 == "2015" ]]; then #The Summer 2020 Ultra Legacy samples for 2015 are named "16MiniAODAPV"; the regular 2016 files do not have the "APV".
        dasYear="RunIISummer20UL16MiniAODAPV"
    else # All other years are straightforward. This trims the first two characters off of the string, so 2017 becomes 17, etc.
        dasYear="RunIISummer20UL${2:2}MiniAOD"
    fi
    
    # Here we build the case structure for each particle, and define "$dasPart", which is used to search DAS.
    # The -a flag is for arrays, the -g flag declares the array globally, allowing us to call it outside of the checkDAS function.
    # The sort -r command sorts the results into reverse order. This is helpful for looping over the datasets and detecting the v2 datasets;
    # the v2 datasset will always print right after the v1 dataset, so by filling the array in reverese, we get the v2 dataset first each time, simplifying future loops.
    # Note that everytime this function is called, $dasDatasets is overwritten.
    case $1 in
        "WW") # W case
            dasPart="BulkGravToWWToWhadWhad"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
        ;;        
        "ZZ") # Z case
            dasPart="BulkGravToZZToZhadZhad"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
        ;;        
        "bb") # b case
            dasPart="ZprimeToBB"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
        ;;
        "HH") # Higgs case
            dasPart="GluGluToBulkGravitonToHHTo4B"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
        ;;
        "tt") # Top case
            dasPart="ZprimeToTT"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
        ;;
        "QCD") # QCD case (Flattened pT samples only exist for 2015 and 2016 at the moment)
            dasPart="QCD_Pt*to*"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )

            declare -a trimmedDatasetsQCD
            for datset in ${dasDatasets[*]}; do # Need to trim the low pT files; want to keep everything above 400pT (so we lowest we keep 300to470)
                if [[ ! "$datset" =~ ("15to30"|"30to50"|"50to80"|"80to120"|"120to170"|"170to300") ]]; then # If dataset is one of these pT ranges, add it to $trimmedDatasetsQCD
                    trimmedDatasetsQCD+=( "$datset" )
                fi
            done
            dasDatasets=( "${trimmedDatasetsQCD[@]}" ) # Update $dasDatasets to have desired pT samples          
        ;;
    esac

    if [[ $2 == "2016" ]]; then # If year = 2016, then the 2015 datasets will be mixed into $dasDatasets; this code removes those datasets.
        declare -a trimmedDatasetsYear
        for datset in ${dasDatasets[*]}; do
            if [[ "$datset" != *"RunIISummer20UL16MiniAODAPV"* ]]; then # If dataset is NOT a 2015 dataset, add the dataset to $trimmedDatasetsYear
                trimmedDatasetsYear+=( "$datset" )
            fi
        done
        dasDatasets=( "${trimmedDatasetsYear[@]}" ) # Update $dasDatasets to have only 2016 samples
    fi

    # Uncomment this if you want to print what datasets are being found by checkDAS
    # for item in ${dasDatasets[@]}; do
    #     echo $item
    # done
}

# For our analysis, there are updated versions of the datasets, with "v2" appended to the end of $dasYear. These datasets are still coming out, so this code will need to be able to check for new v2 datasets.

# This loop will create the text files, search for the datasets on DAS, check for version, and fill the text files accordingly.
for dat in ${myDatatypes[*]}; do #Loop over mc and data
    if [[ $dat == "data" ]] ; then continue; fi #skips the loop for data, not implemented yet
    echo "${PURP}Beginning $dat${NC}..."
    for yr in ${myYears[*]}; do #Loop over years
        echo "${GRN}Beginning $yr${NC}..."
        # Check if file exists, if so delete 
        fileToWrite="../samples/${dat}_${yr}.txt"
        echo "${YEL}Writing samples to:${NC} $fileToWrite"
        if [ -f $fileToWrite ] ; then rm $fileToWrite; fi
        for part in ${myParticles[*]}; do #Loop over particles
            echo "${CYAN}Beginning $part${NC}..."
            #These help keep track of how many datasets still need a version 2.
            echo "#$part" >> $fileToWrite
            v2Counter=0
            v1Counter=0
            v2Flag=false # Boolean flag
            v1Expected= # Empty strings to fill and compare later...
            v1Actual= # ...to help keep an eye out for errors
            checkDAS $part $yr $dat # Check DAS for every v1 and v2 dataset for this particle+year+datatype, load results into $dasDatasets array
            for datset in ${dasDatasets[*]}; do # Loop over DAS search results. Now we are checking each DAS dataset individually.
                if [[ "$datset" == *"${dasYear}v2"* ]]; then # Trigers if the dataset is a v2 dataset
                    echo $datset >> $fileToWrite # Write the dataset to the samples file
                    ((v2Counter++)) # This increments v2counter
                    v2Flag=true # If this is a v2 dataset, then the next dataset in the loop SHOULD be the v1 version of this dataset--so we trigger this flag.
                    v1Expected=${datset/"${dasYear}v2"*/${dasYear}} # This string SHOULD be the beginning of the dataset name of the v1 version of this dataset (UNNECESSARY IF YOU TRUST DAS, WHICH YOU SHOULDN'T)
                elif $v2Flag; then # This will only trigger if the previous dataset was a v2. This will reset the v2flag and skip the v1 dataset, or throw an error if the names don't match
                    v2Flag=false # Reset the v2flag
                    v1Actual=${datset/"${dasYear}"*/${dasYear}} # This string is the ACTUAL beginning of the v1 dataset name. Since particle, year, and datatype must be the same, this is essentially checking that the mass points match.
                    if [[ "${v1Actual}" != "${v1Expected}" ]]; then # This triggers if the supposed v1 dataset doesn't match the v2--this shouldn't ever trigger unless something weird happens with the future DAS dataset names.
                        echo "${RED}ERROR: THE BEGINNING OF v2 AND v1 DATASET NAMES DO NOT MATCH.${NC}"
                        echo "${YEL}v1 dataset expected to be named:${NC} ${v1Expected}"
                        echo "${YEL}But it is actually named:${NC} ${v1Actual}"
                        echo "${RED}MOST LIKELY THE MASS POINTS DO NOT MATCH."
                        exit 1
                    fi
                else # If this triggers, then this dataset is a v1, and there is for sure not a v2 version of it.
                    echo $datset >> $fileToWrite # Write the dataset to the samples file                            
                    ((v1Counter++)) # This increments v1counter                    
                fi
            done
            echo "For the /${CYAN}${dasPart}${NC}*/${GRN}${dasYear}${NC}*/${PURP}MINIAODSIM${NC} datasets, ${BLUE}there are ${v2Counter} v2 datasets and ${v1Counter} v1 datasets${NC}."
        done
        echo "${YEL}Checkout your new list of files at:${NC} $fileToWrite"
    done
done
#Undo the alias used for this script
unalias echo