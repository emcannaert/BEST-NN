#!/bin/bash
#=========================================================================================
# fillSamples.sh -------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script lives in the BEST/scripts directory, but should be executed through the symbolic link in the BEST/preprocess/crab directory.
# This script fills text files in the BEST/samples directory. If it is moved, then some paths will need to be updated for it to function correctly.
# This script takes arguments for particle, year, and datatype to create/fill the sample files with mass points,dataset names from DAS, using dasgoclient.
# This script is specific to calling datasets for the Summer 2020 Ultra Legacy samples submitted by the UCD BEST team, for the purpose of training BEST. But it can be modified to search for other datasets! 
# This script also checks for and keeps track of a version two, or "v2", for each dataset, as these updated datasets are still being produced as of writing this code.

######################################### NOTES TO SELF ############################
# Implement data
# Implement finding mass points that are closest to the list
# Implment skipping finding DAS file?
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
# This alias makes the script simpler, as '-e' is needed to print color. This is undone by 'unalias' at the end of the code.
shopt -s expand_aliases
alias echo='echo -e'

# Check that user provided valid amount of options, exit if not. The only valid inputs would have 0 or 6 options/arguments.
if [[ $# != 0 ]] && [[ $# != 6 ]]; then 
    echo "${YEL}WOAH, slow down there friend!${NC} Your command line contains $# options/arguments!"
    echo "Please pass options and arguments as:"
    echo
    echo "./fillSamples.sh ${CYAN}-p \"<particle 1> <particle 2>...\" ${GRN}-y \"<year 1> <year 2>...\" ${PURP}-d <datatype>${NC}"
    echo
    echo "${CYAN}Particle arguments: ${BLUE}all${NC}, ${YEL}or${NC} any combination of ${CYAN}QCD, HH, WW, ZZ, tt, bb${NC}"
    echo "${GRN}Year arguments: ${BLUE}all${NC}, ${YEL}or${NC} any combination of ${GRN}2016_APV, 2016, 2017, 2018${NC}"
    echo "${PURP}Datatype arguments: ${BLUE}all${NC} ${YEL}or${NC} ${PURP}mc${NC} ${YEL}or${NC} ${PURP}data${NC}"
    echo "${YEL}Note that for the ${GRN}year${YEL} arguments, ${GRN}2016_APV${YEL} is a special case. It corresponds to the 2015 datasets, but in DAS it us under 2016 with APV in the dataset name."
    echo
    echo "All options and arguments are case-sensitive, and all options-argument pairs can be executed in any order."
    echo "${YEL}Example:${NC} ./fillSamples.sh ${PURP}-d mc ${GRN}-y ${BLUE}all ${CYAN}-p HH${NC}"
    echo
    echo "Quotes are necessary when passing multiple arguments for one option."
    echo "${YEL}Example:${NC} ./fillSamples.sh ${GRN}-y \"2016_APV 2016 2017\" ${PURP}-d data ${CYAN}-p \"HH WW\"${NC}"    
    echo
    echo "Simply running the script without any options or arguments selects the '${BLUE}all${NC}' argument for ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype.${NC}"
    echo "${YEL}Example:${NC} ./fillSamples.sh"
    exit 1
fi

# Declare the full list of valid arguments for each option
declare -a allParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")
declare -a allYears=("2016_APV" "2016" "2017" "2018")
declare -a allDatatypes=("mc" "data")
# Declare initial arrays to fill with user chosen arguments later
declare -a myParticles
declare -a myYears
declare -a myDatatypes

# This is where the options and arguments are parsed in.
if [[ $# == 0 ]]; then # Default case, sets up to update everything.
    echo "Default behavior triggered. ${BLUE}All${NC} samples for each ${CYAN}particle${NC}, ${GRN}year${NC}, and ${PURP}datatype${NC}, will be generated."
    myParticles=${allParticles[*]}
    myYears=${allYears[*]}
    myDatatypes=${allDatatypes[*]}
else # Specific cases, sets up to update specfic datasets.
    while getopts :p:y:d: opt; do
        case $opt in
            p)  #T he -p option, for particles. Either fills "all" or the specificly chosen arguments.
                if [[ $OPTARG == "all" ]]; then
                    myParticles=${allParticles[*]}
                else 
                    for part in $OPTARG; do    
                        if [[ ${allParticles[*]} =~ $part ]]; then # Check for valid arguments, then fills array.
                            myParticles+=($part)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${CYAN}$opt${NC}: $part"
                            echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${CYAN}${allParticles[*]}${NC}"
                            echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
                            echo "${RED}Exiting without creating samples...${NC}"
                            exit 1
                        fi
                    done
                fi
            ;;
            y)  # The -y option, for years. Either fills "all" or the specificly chosen arguments.
                if [[ $OPTARG == "all" ]]; then
                    myYears=${allYears[*]}
                else
                    for yr in $OPTARG; do
                        if [[ ${allYears[*]} =~ $yr ]]; then # Check for valid arguments, then fills array.
                            myYears+=($yr)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${GRN}$opt${NC}: $yr"
                            echo "Please choose '${BLUE}all${NC}', or the case-sensitive arguments: ${GRN}${allYears[*]}${NC}"
                            echo "${YEL}Run script without any options to see usage:${NC} ./fillSamples.sh"
                            echo "${RED}Exiting without creating samples...${NC}"
                            exit 1
                        fi
                    done
                fi
            ;;
            d)  # The -d option, for datatype. Either fills "all" or the specificly chosen arguments.
                if [[ $OPTARG == "all" ]]; then
                    myDatatypes=${allDatatypes[*]}
                else
                    for dat in $OPTARG; do
                        if [[ ${allDatatypes[*]} =~ $dat ]]; then # Check for valid arguments, then fills array.
                            myDatatypes+=($dat)
                        else # Invalid arguments trigger error message
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
fi

echo "${CYAN}Particle(s)${NC} selected: ${CYAN}${myParticles[*]}${NC}"
echo "${GRN}Year(s)${NC} selected: ${GRN}${myYears[*]}${NC}"
echo "${PURP}Datatype(s)${NC} selected: ${PURP}${myDatatypes[*]}${NC}"
echo
echo "${BLUE}Initiating DAS search. Checking voms cms proxy...${NC}"

# This checks for a voms cms proxy that will last longer than 60 minutes, and has the user create a new one if not
if [[ $(voms-proxy-info -timeleft) > 3600 ]] && [[ $(voms-proxy-info -vo) == "cms" ]]; then
    echo "${GRN}Valid proxy confirmed!${NC}"
    echo
else
    echo "${YEL}Error: Proxy either doesn't exist or will expire soon.${NC} Initializing new proxy..."
    voms-proxy-init --valid 192:00 -voms cms
    echo
fi

# These 21 mass points were originally used when submitting the GridPacks:
# declare -a massPoints=("500" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000" "2500" "3000" "3500" "4000" "4500" "5000" "5500" "6000" "6500" "7000" "7500" "8000")

# Here we define a function to call dasgoclient to search DAS, and assigns result to array of strings to "$dasDatasets".
# This also saves the mass point for each dataset that DAS finds to an array of strings "$dasMass".
# Also uses "$dasMass" to create unique crab job names for each dataset, filling the array of strings "$crabNames".
# This also defines two helpful strings, "dasYear" and "dasPart", and trims away unwanted datasets.
checkDASDatasets(){ ################### Takes inputs as: "checkDASDatasets particle year datatype" #########################
    # To search for our datasets on DAS, we need to manipulate the input data a bit. This is specific to our current analysis but can be modified for other analyses.
    # Here we define "$dasYear", which is used to search DAS.
    if [[ $2 == "2016_APV" ]]; then #The Summer 2020 Ultra Legacy samples for 2016_APV (2015) are named "16MiniAODAPV"; the regular 2016 files do not have the "APV".
        dasYear="RunIISummer20UL16MiniAODAPV"
    else # All other years are straightforward. This trims the first two characters off of the string, so 2017 becomes 17, etc.
        dasYear="RunIISummer20UL${2:2}MiniAOD"
    fi

    # Here we build the case structure for each particle, and define "$dasPart", which is used to search DAS.
    # The -a flag is for arrays, the -g flag declares the bash array globally, allowing us to call it outside of the checkDASDatasets function.
    # The sort -r command sorts the results from dasgoclient into reverse order. This is helpful for looping over the datasets and detecting the v2 datasets;
    # The v2 dataset will always print right after the v1 dataset, so by filling the array in reverese, we get the v2 dataset first each time, simplifying the trimming process at the end of this function.
    # Note that everytime this function is called, $dasDatasets, $crabNames, $dasMass, $dasYear, and $dasPart are overwritten.
    # $dasMass, $crabNames, and $dasDatasets fill the sample files, and are used by the python script buildDict.py to submit crab jobs.
    # The case structure handles all the things that are unique to the different particle sets. More trimming of the data occurs after the case structure.
    declare -ag crabNames=()
    declare -ag dasMass=()
    case $1 in
        "WW") # W case
            dasPart="GravToWW"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/Bulk${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
            for datset in ${dasDatasets[@]}; do # Fill $dasMass and $crabNames
                trimstring1=${datset#*"_M-"} # Trims the front of string
                trimstring2=${trimstring1%"_Tune"*} # Trims back of string; now $trimstring2 is the mass point of the dataset
                dasMass+=( "$trimstring2" ) # This is the mass point in GeV
                crabNames+=( "GravitonWW_${trimstring2}GeV_trees" ) # This will be the name of the crab project directory 
            done 
        ;;        
        "ZZ") # Z case
            dasPart="GravToZZ"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/Bulk${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
            for datset in ${dasDatasets[@]}; do # Fill $dasMass and $crabNames
                trimstring1=${datset#*"_M-"} # Trims the front of string
                trimstring2=${trimstring1%"_Tune"*} # Trims back of string; now $trimstring2 is the mass point of the dataset
                dasMass+=( "$trimstring2" ) # This is the mass point in GeV
                crabNames+=( "GravitonZZ_${trimstring2}GeV_trees" ) # This will be the name of the crab project directory 
            done        
        ;;        
        "bb") # b case
            dasPart="ZprimeToBB"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
            for datset in ${dasDatasets[@]}; do # Fill $dasMass and $crabNames
                trimstring1=${datset#*"_M-"} # Trims the front of string
                trimstring2=${trimstring1%"_Tune"*} # Trims back of string; now $trimstring2 is the mass point of the dataset
                dasMass+=( "$trimstring2" ) # This is the mass point in GeV                
                crabNames+=( "ZprimeBB_${trimstring2}GeV_trees" ) # This will be the name of the crab project directory 
            done        
        ;;
        "HH") # Higgs case
            dasPart="GravitonToHH"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/GluGluToBulk${dasPart}To4B*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
            for datset in ${dasDatasets[@]}; do # Fill $dasMass and $crabNames
                if [[ "$datset" =~ "60000" ]]; then
                    crabNames+=( "GravitonHH_6000GeV_trees" ) # Special case where the mc dataset for 6000Gev was accidentally named 60000Gev in DAS. The data is generated correctly at 6000GeV, so we just need to rename the crab directory.
                    dasMass+=( "6000" ) # This is the mass point in GeV                    
                else
                    trimstring1=${datset#*"_M-"} # Trims the front of string
                    trimstring2=${trimstring1%"_narrow"*} # Trims back of string; now $trimstring2 is the mass point of the dataset
                    dasMass+=( "$trimstring2" ) # This is the mass point in GeV
                    crabNames+=( "GravitonHH_${trimstring2}GeV_trees" ) # This will be the name of the crab project directory 
                fi
            done   
        ;;
        "tt") # Top case
            dasPart="ZprimeToTT"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
            for datset in ${dasDatasets[@]}; do # Fill $dasMass and $crabNames
                trimstring1=${datset#*"_M"} # Trims the front of string
                trimstring2=${trimstring1%"_W"*} # Trims back of string; now $trimstring2 is the mass point of the dataset
                dasMass+=( "$trimstring2" ) # This is the mass point in GeV
                crabNames+=( "ZprimeTT_${trimstring2}GeV_trees" ) # This will be the name of the crab project directory 
            done        
        ;;
        "QCD") # QCD case (Flattened pT samples only exist for 2015 and 2016 at the moment)
            dasPart="QCD_Pt"
            declare -ag dasDatasets=( $(dasgoclient -query="dataset dataset=/${dasPart}*to*/${dasYear}*-106X*/MINIAODSIM datatype=$3" | sort -r) )
            declare -a trimmedDatasetsQCD
            for datset in ${dasDatasets[*]}; do # Need to trim the low pT files; want to keep everything above 500pT (so we lowest we keep is 470to600)
                if [[ ! "$datset" =~ ("15to30"|"30to50"|"50to80"|"80to120"|"120to170"|"170to300"|"300to470") ]]; then # If dataset is not one of these pT ranges, add it to $trimmedDatasetsQCD
                    trimmedDatasetsQCD+=( "$datset" )
                fi
            done
            dasDatasets=( "${trimmedDatasetsQCD[@]}" ) # Update $dasDatasets to have only the desired pT samples     

            for datset in ${dasDatasets[@]}; do # Fill $dasMass and $crabNames
                if [[ "$datset" =~ "Flat" ]]; then
                    crabNames+=( "QCD_Flat_Pt_trees" ) # Special case where the Pt is flattened across the whole range. Compare to the Pt binned data.
                    dasMass+=( "Flat" ) # Pt range
            else
                    trimstring1=${datset%"_Tune"*} # Trims back of string; now $trimstring is the Pt range of the dataset, with a slash in front, like: "/QCD_Pt_*to*"
                    crabNames+=( "${trimstring1:1}_trees" ) # This will be the name of the crab project directory. The :1 removes the first character, which is the slash. 
                    trimstring2=${trimstring1#*"Pt_"} # Trims the front of string
                    dasMass+=( "${trimstring2}" ) # This is the pT range in GeV, like "470to600"
                fi
            done
        ;;
    esac

    # At this point, $dasMass, $crabNames, and $dasDatasets should exactly match each other, and each be arrays correctly filled with particle datasets. 
    # However, we still need to do two things to our dataset arrays:

    # If year = 2016, then the 2016_APV (2015) datasets will be mixed into our arrays; this code removes those datasets:
    if [[ $2 == "2016" ]]; then 
        declare -a trimmedDatasetsYear=()
        declare -a trimmedDasMassYear=()
        declare -a trimmedCrabNamesYear=()
        for i in ${!dasDatasets[*]}; do
            if [[ "${dasDatasets[$i]}" != *"RunIISummer20UL16MiniAODAPV"* ]]; then # If dataset is NOT a 2015 dataset, add the dataset to trimmed arrays
                trimmedDatasetsYear+=( "${dasDatasets[$i]}" )
                trimmedDasMassYear+=( "${dasMass[$i]}" ) 
                trimmedCrabNamesYear+=( "${crabNames[$i]}" )
            fi
        done
        dasDatasets=( "${trimmedDatasetsYear[@]}" ) # Update $dasDatasets to have only 2016 samples
        dasMass=( "${trimmedDasMassYear[@]}" ) # Update $dasMass to have only 2016 samples
        crabNames=( "${trimmedCrabNamesYear[@]}" ) # Update $crabNames to have only 2016 samples
    fi

    # For our analysis, there are updated versions of the datasets, with "v2" appended to the end of $dasYear. These datasets are still coming out, so this code will need to be able to check for new v2 datasets.
    # Now we trim the away the v1 datasets if they have a v2; this process is identical across all samples:
    v2Counter=0 # These two counters help us report how many datasets still need a v2
    v1Counter=0
    declare -a trimmedDatasetsv2=()
    declare -a trimmedDasMassv2=()
    declare -a trimmedCrabNamesv2=()
    v2Flag=false # Flag that triggers if we find a v2 dataset
    v1Expected=() # Empty strings to fill and compare later...
    v1Actual=() # ...to help keep an eye out for errors
    for i in ${!dasDatasets[*]}; do 
        if [[ "${dasDatasets[$i]}" == *"${dasYear}v2"* ]]; then # Triggers if the dataset is a v2 dataset, adds it to trimmed arrays
            trimmedDatasetsv2+=( "${dasDatasets[$i]}" )
            trimmedDasMassv2+=( "${dasMass[$i]}" )
            trimmedCrabNamesv2+=( "${crabNames[$i]}" )            
            ((v2Counter++)) # Increments v2counter
            v2Flag=true # If this is a v2 dataset, then the next dataset in the loop SHOULD be the v1 version of this dataset--so we trigger this flag.
            v1Expected=${dasDatasets[i]/"${dasYear}v2"*/${dasYear}} # This string SHOULD be the beginning of the dataset name of the v1 version of this dataset (UNNECESSARY IF YOU TRUST DAS, WHICH YOU SHOULDN'T)
        
        elif $v2Flag; then # This will only trigger if the previous dataset in the loop was a v2. This will reset the v2flag and skip the v1 dataset, or throw an error if the names don't match
            v2Flag=false # Reset the v2flag
            v1Actual=${dasDatasets[i]/"${dasYear}"*/${dasYear}} # This string is the ACTUAL beginning of the v1 dataset name. Since particle, year, and datatype must be the same, this is essentially checking that the mass points match.
            if [[ "${v1Actual}" != "${v1Expected}" ]]; then # This triggers if the supposed v1 dataset doesn't match the v2--this shouldn't ever trigger unless something weird happens with the future DAS dataset names.
                echo "${RED}ERROR: THE BEGINNING OF v2 AND v1 DATASET NAMES DO NOT MATCH.${NC}"
                echo "${YEL}v1 dataset expected to be named:${NC} ${v1Expected}"
                echo "${YEL}But it is actually named:${NC} ${v1Actual}"
                echo "${RED}MOST LIKELY THE MASS POINTS DO NOT MATCH."
                exit 1
            fi

        else # Triggers if the dataset is a v1 without a v2 yet; adds it to new array
            trimmedDatasetsv2+=( "${dasDatasets[$i]}" )
            trimmedDasMassv2+=( "${dasMass[$i]}" )
            trimmedCrabNamesv2+=( "${crabNames[$i]}" )   
            ((v1Counter++)) # Increments v1counter                    
        fi
    done
    dasDatasets=( "${trimmedDatasetsv2[@]}" ) # Update $dasDatasets
    dasMass=( "${trimmedDasMassv2[@]}" ) # Update $dasMass
    crabNames=( "${trimmedCrabNamesv2[@]}" ) # Update $crabNames 

    # At this point, the datasets are ready to go--the particle is correct, the APV (2015) datasets have been removed from the standard 2016 datasets, and the only v1 datasets around are the ones missing v2 datasets.
}

# This function checks DAS for files for each dataset. Since each dataset needs to check DAS individually, this part can take quite a while, but makes the code much simpler than checkDASDatasets.
# dasgoclient will return an array of files names, but we want each file to have about the same events. So we check DAS twice for each dataset, once for the names, once for the list of nevents.
checkDASFiles(){ ################## Takes input as: "checkDASFiles dataset"
    declare -a dasAllFiles=( $(dasgoclient -query="file dataset=$1") )
    declare -ag dasAllEvents=( $(dasgoclient -query="file dataset=$1 | grep file.nevents" ) )
    dasFile=()
    dasEvent=0
    for k in ${!dasAllFiles[*]}; do # Find file with the most events
        if (( ${dasAllEvents[$k]} > $dasEvent )); then
            dasFile=( "${dasAllFiles[$k]}" )
            dasEvent=( "${dasAllEvents[$k]}" )
        fi
    done 
}

# This loop will create the text files, search for the datasets on DAS, check for version, and fill the text files accordingly.
for dat in ${myDatatypes[*]}; do # Loop over mc and data
    if [[ $dat == "data" ]] ; then break; fi #skips the loop for data, not implemented yet
    echo "${PURP}Beginning $dat...${NC}"

    for yr in ${myYears[*]}; do # Loop over years
        echo "${GRN}Beginning $yr...${NC}"

        # Define file names
        fileToWrite="../../samples/${dat}_${yr}.txt"
        filePrevious="../../samples/previous/${dat}_${yr}.txt"

        if [[ -f "$fileToWrite" ]] ; then # Triggers if there is already a set of sample files
            if [[ -f "$filePrevious" ]]; then # If a previous set of samples already exists, delete it
                echo "${YEL}Deleting old previous sample file at:${NC} $filePrevious"        
                rm $filePrevious 
            fi 
            echo "${YEL}Renaming current sample file from:${NC} $fileToWrite ${YEL}to the new previous sample file:${NC} $filePrevious" 
            mkdir -p ../../samples/previous # Makes sure the previous directory exists    
            mv $fileToWrite $filePrevious # Shifts the current samples to now be previous samples
        elif [[ -f "$filePrevious" ]]; then
            echo "${YEL}No current sample file detected. Will use, but not change, previous sample file: $filePrevious"
        else
            echo "${YEL}No current or previous sample files detected. Will build fresh ${GRN}$yr ${PURP}$dat ${YEL}sample files from only the user inputted${NC} ${CYAN}particle ${YEL}selections. This might take a few minutes."
        fi

        echo "${YEL}Writing samples to:${NC} $fileToWrite"
        echo "# Mass Point, Crab Directory, \tDataset Name, \t File With Most Events" >> $fileToWrite # Labels the particle sections in sample file
        for part in ${allParticles[*]}; do # Loop over particles
            echo "${CYAN}Beginning $part...${NC}"
            echo "#$part" >> $fileToWrite # Labels the particle sections in sample file

            if [[ -f "$filePrevious" ]]; then # If a previous sample file exists, read in the corresponding particle section of data and then use it to compare with current datasets
                targetSection=false
                declare -a dasPrevious=()
                declare -a massPrevious=()
                declare -a crabPrevious=()
                declare -a dasFilePrevious=()
                file=$filePrevious
                OLDIFS=$IFS # Preserve the old IFS to reinstate it later
                IFS=',' # This lets us read in the comma separated sample files
                while read first second third fourth; do # Reads the previous sample file, stores values to arrays
                    if [[ "$first" == "#$part" ]]; then # Triggers when reaching the beginning of the relevant lines for this $part loop
                        targetSection=true
                    elif [[ "$targetSection" == true ]]; then
                        if [[ ${allParticles[*]} =~ "${first:1}" ]]; then break; fi # Ends while loop when the next section of datasets begins ######### MIGHT NEED AN OR STATEMENT HERE FOR QCD
                        massPrevious+=( "$first" )
                        crabPrevious+=( "$second" )
                        dasPrevious+=( "$third" )
                        dasFilePrevious+=( "$fourth" )
                    fi        
                done < "$filePrevious"
                IFS=$OLDIFS # Resets to \n so the rest of the code works

                if [[ ${myParticles[*]} =~ $part ]]; then # Checks if user specified this particle. If so, check DAS for updates to the list of datasets.
                    checkDASDatasets $part $yr $dat # Check DAS for user submitted particle/year/datatype set, loads several arrays/strings/counters, prints out v1 vs. v2 dataset counters
                    j=0 # Separate iterator that runs through the Previous arrays (if there is a new v1 dataset found by check DAS, then the size of $dasDatasets and $dasPrevious will not match)
                    newcounter=0 # Keeps track of how many new files are found
                    
                    for i in ${!dasDatasets[*]}; do # Begin loop over dataset results
                        if [[ ${dasPrevious[$j]} =~ ${dasDatasets[$i]} ]]; then # Checks to see if new dataset is already in the Previous file. Results should be the same order, minus any new files.                        
                            echo "${massPrevious[$j]},${crabPrevious[$j]},${dasPrevious[$j]},${dasFilePrevious[$j]}" >> $fileToWrite # Write the dataset to the samples file; using Previous results saves time checking DAS for individual files
                            ((j++)) # Increments Previous array index
    
                        else # Triggers if there is a new dataset; that is, DAS found a dataset that is not contained in the Previous sample file. Two cases: a v1 got updated to a v2, or a completely new mass point was released (would be a v1).
                            ((newcounter++)) # Increment new files counter
                            if [[ "${dasDatasets[$i]}" == *"${dasYear}v2"* ]]; then # Triggers if the new dataset is a v2 dataset; need to update the sample file to replace the old v1 dataset with the new v2 dataset
                                echo "${YEL}NEW v2 DATASET DETECTED:${NC} ${dasDatasets[$i]}" 

                                # The current $dasDatasets element should be the v2 version of the current $dasPrevious v1 element. We check to make sure:
                                v1ExpectedNew=${dasDatasets[$i]/"${dasYear}v2"*/${dasYear}} # This string SHOULD be the beginning of the dataset name of the v1 version of this dataset 
                                v1ActualPrevious=${dasPrevious[$j]/"${dasYear}"*/${dasYear}} # This string is the ACTUAL beginning of the v1 dataset name, contained in the Previous array.
                        
                                if [[ "${v1ActualPrevious}" != "${v1ExpectedNew}" ]]; then # This triggers if the supposed old v1 dataset doesn't match the new v2--this shouldn't ever trigger unless something went wrong with indexing or DAS names
                                    echo "${RED}INDEXING ERROR: NEW v2 DATASET DETECTED, BUT CORRESPONDING v1 DATASET NAME DOES NOT MATCH PREVIOUS SAMPLE FILE'S v1 DATASET NAME.${NC}"
                                    echo "${YEL}v1 dataset expected to be named:${NC} ${v1ExpectedNew}"
                                    echo "${YEL}But it is actually named:${NC} ${v1ActualPrevious}"
                                    echo "${RED}EXAMINE PREVIOUS SAMPLE FILE FOR ERRORS, OR CHECK DAS DATASET NAMES FOR INCONSISTENCIES.${NC}"
                                    echo "${RED}INCORRECTLY CHANGING THE DATASETS IN THE SAMPLE FILES COULD CAUSE THIS.${NC}"
                                    echo "${RED}THIS WOULD BE ALSO BE TRIGGERED IF THERE IS A NEW v2 DATASET BUT THERE IS NO CORRESPONDING v1 DATASET. TO RESET, SIMPLY DELETE ALL SAMPLE FILES, AND RUN ./fillSamples.sh WITH NO OPTIONS. IF DAS HAS A v2 DATASET BUT NOT A v1, THEN AN ERROR WOULD HAVE BEEN THROWN EARLIER IN THE CODE. ${NC}"
                                    exit 1
                        
                                else # Triggers if the new v2 dataset corresponds to the old v1 dataset; this is what SHOULD trigger each time
                                    checkDASFiles ${dasDatasets[$i]} # Checks DAS for files associated with this dataset. Loads $dasFile and $dasEvent strings.
                                    echo "${dasMass[$i]},${crabNames[$i]},${dasDatasets[$i]},${dasFile}" >> $fileToWrite # Write new dataset to samples file
                                    ((j++)) # Increments Previous array index, skipping the v1 version of this dataset          
                                fi 

                            else # Triggers if the new dataset is a v1 dataset. We DO NOT increment j, since the dasPrevious element should currently be the next element of dasDatasets in the loop.
                                echo "${YEL}NEW v1 DATASET DETECTED:${NC} ${dasDatasets[$i]}"     
                                checkDASFiles ${dasDatasets[$i]} # Checks DAS for files associated with this dataset. Loads $dasFile and $dasEvent strings.
                                echo "${dasMass[$i]},${crabNames[$i]},${dasDatasets[$i]},${dasFile}" >> $fileToWrite # Write new dataset to samples file

                            fi
                        fi
                    done
                    echo "${YEL}$newcounter new files found.${NC}"
                    echo "${BLUE}For the${NC} /${CYAN}${dasPart}${NC}*/${GRN}${dasYear}${NC}*/${PURP}MINIAODSIM ${BLUE}datasets, wrote ${YEL}${v2Counter} v2 ${BLUE}datasets and ${YEL}${v1Counter} v1 ${BLUE}datasets to${NC} ${fileToWrite}"
                else # If the user did not specify the particle, and a Previous sample file exists, simply copy the previous file for that particle.
                    echo "${CYAN}Copying $part...${NC}"        
                    for j in ${!dasPrevious[*]}; do
                        echo "${massPrevious[$j]},${crabPrevious[$j]},${dasPrevious[$j]},${dasFilePrevious[$j]}" >> $fileToWrite # Write the dataset to the samples file; using Previous results saves time checking DAS for individual files
                    done
                fi     
            else # Triggers if there is not a previous sample file
                if [[ ${myParticles[*]} =~ $part ]]; then # Checks if user specified this particle. If so, check DAS for datasets and fill sample file.
                    checkDASDatasets $part $yr $dat # Check DAS for user submitted particle/year/datatype set, loads several arrays/strings/counters, prints out v1 vs. v2 dataset counters
                    for i in ${!dasDatasets[*]}; do # Begin loop over dataset results
                        checkDASFiles ${dasDatasets[$i]} # Checks DAS for files associated with this dataset. Loads $dasFile and $dasEvent strings.
                        echo "${dasMass[$i]},${crabNames[$i]},${dasDatasets[$i]},${dasFile}" >> $fileToWrite # Write new dataset to samples file
                    done
                    echo "${BLUE}For the${NC} /${CYAN}${dasPart}${NC}*/${GRN}${dasYear}${NC}*/${PURP}MINIAODSIM ${BLUE}datasets, wrote ${YEL}${v2Counter} v2 ${BLUE}datasets and ${YEL}${v1Counter} v1 ${BLUE}datasets to${NC} ${fileToWrite}"
                else
                    echo "${CYAN}Skipping $part...${NC}"
                fi
            fi
        done
        echo "${YEL}Checkout your new list of files at:${NC} $fileToWrite"
    done
done

# Build dictionary from sample files
python buildDict.py

# Undo the alias used for this script
unalias echo