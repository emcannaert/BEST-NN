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

################################## NOTES TO SELF ##################################
# Implement data

###(NOTE: 2015 = 2016_APV)###
# As of Dec. 10, 2021:
#   Missing Mass Points:
#       ZZ:  2018: 1000
#       QCD: 2017: Flat
#       tt:  2017,2018: 400, 500, 600, 700, 800, 900, 1000; All years: 5000, 5500, 6000, 6500, 7000, 7500, 8000
#   Dataset Versions:
#       2015: All tt and QCD datasets are v2, the rest is v1.
#       2016: All datasets are v2
#       2017: All datasets are v2, except for one extra tt dataset (detailed below)
#       2018: All datasets are v2 (NOTE: No v1 dataset exists for QCD Flat)
#   Notes:
#       HH: 60000 mass point instead of 6000 on DAS for HH for all years (checked this, the mass point is correctly 6000, the name is just wrong)
#       tt: Mass points on DAS not in the 21 mass points given: (2015 and 2016: 400, 700, 900) <- The script finds all samples, regardless of relation to the 21 mass points requested
#       tt: Using extra tt samples that were not originally requested. Can be identified by a capital "P" in dataset name (/ZPrimeToTT... instead of /ZprimetoTT...)
#           Gives 2 extra v2 datasets per mass point (17 from 400 to 4500, 34 total each year) at 30% and 10% width, except that (2017: M900_W270) is v1


#==================================================================================
# Setup ///////////////////////////////////////////////////////////////////////////
#==================================================================================

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
                    for particle in $OPTARG; do    
                        if [[ ${allParticles[*]} =~ $particle ]]; then # Check for valid arguments, then fills array.
                            myParticles+=($particle)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${CYAN}$opt${NC}: $particle"
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
                    for year in $OPTARG; do
                        if [[ ${allYears[*]} =~ $year ]]; then # Check for valid arguments, then fills array.
                            myYears+=($year)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${GRN}$opt${NC}: $year"
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
                    for datatype in $OPTARG; do
                        if [[ ${allDatatypes[*]} =~ $datatype ]]; then # Check for valid arguments, then fills array.
                            myDatatypes+=($datatype)
                        else # Invalid arguments trigger error message
                            echo "${YEL}Error:${NC} Invalid argument for ${PURP}$opt${NC}: $datatype"
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
if [[ ! $(voms-proxy-info -timeleft) > 3600 ]] || [[ $(voms-proxy-info -vo) != "cms" ]]; then
    echo "${YEL}Error: Proxy either doesn't exist or will expire soon. Initializing new proxy...${NC}"
    voms-proxy-init --valid 192:00 -voms cms
    if [[ ! $(voms-proxy-info -timeleft) > 3600 ]] || [[ $(voms-proxy-info -vo) != "cms" ]]; then exit 1; fi # Exit if user failed to create proxy
fi
echo "${GRN}Valid proxy confirmed!${NC}"

# Declare $dasFront, an associative array of strings used to search DAS and trim strings:
declare -Ag dasFront=(  ["HH"]="GluGluToBulkGravitonToHHTo4B_M-"    ["WW"]="BulkGravToWWToWhadWhad_narrow_M-"   ["ZZ"]="BulkGravToZZToZhadZhad_narrow_M-" 
                        ["tt"]="Z*rimeToTT_M"                       ["bb"]="ZprimeToBB_narrow_M-"               ["QCD"]="QCD_Pt_" ) 
# Declare $dasBack, an associative array of strings used to search DAS and trim strings:
declare -Ag dasBack=( ["HH"]="_narrow" ["WW"]="_Tune" ["ZZ"]="_Tune" ["tt"]="_Tune" ["bb"]="_Tune" ["QCD"]="_Tune" )
# The mass point for each sample will the substring in $dasResults that is between $dasFront and $dasBack; the above arrays will also be used to isolate the mass points.

# Declare $crabTemplate, an associative array of strings used to build the crab directory names:
declare -Ag crabTemplate=(  ["HH"]="GravitonHH_M_MASSGeV_trees" ["WW"]="GravitonWW_M_MASSGeV_trees" ["ZZ"]="GravitonZZ_M_MASSGeV_trees" 
                            ["tt"]="ZprimeTT_M_MASSWIDTH_trees"   ["bb"]="ZprimeBB_M_MASSGeV_trees"   ["QCD"]="QCD_Pt_MASS_trees" )

# These 21 mass points were originally used when submitting the GridPacks:
# declare -a massPoints=("500" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000" "2500" "3000" "3500" "4000" "4500" "5000" "5500" "6000" "6500" "7000" "7500" "8000")

#==================================================================================
# Define Functions ////////////////////////////////////////////////////////////////
#==================================================================================

# Here we define a function to call dasgoclient to search DAS, and assign the result to an array of strings to "$dasResults".
# If the datasets pass certain criteria, the datasets in this array are added to an associative array called $dasDatasets.
# This function also saves the mass point for each dataset that passes the criteria to an array of strings "$dasMasses", and uses these masses as keys for $dasDatasets and $crabNames.
# It also uses "$dasMasses" to create unique crab job names for each dataset, filling the associative array of strings "$crabNames".
# These arrays will be comma separated and used to fill the sample text files (like samples/mc_2017.txt) 
# This also defines a helpful string "$dasYear", used to search DAS.
checkDASDatasets(){ ################### Takes inputs as: "checkDASDatasets particle year datatype" #########################
    # For readability, assign function inputs to local variables:
    local particle=$1
    local year=$2
    local datatype=$3
    # To search for our datasets on DAS, we need to manipulate the input data a bit. This is specific to our current analysis but can be modified for other analyses.

    # Here we define "$dasYear", which is used to search DAS.
    # The Summer 2020 Ultra Legacy samples for 2016_APV (2015) are named "16MiniAODAPV"; the regular 2016 files do not have the "APV".
    if [[ $year == "2016_APV" ]]; then  dasYear="RunIISummer20UL16MiniAODAPV"
    # All other years are straightforward. This trims the first two characters off of the year string, so 2017 becomes 17, etc.
    else                                dasYear="RunIISummer20UL${year:2}MiniAOD"; fi
    
    # Clear and declare associative arrays to fill later with DAS search results:
    # The -A flag is for associative arrays, the -g flag declares the bash array globally, allowing us to call it outside of the checkDASDatasets function.
    unset crabNames # For some reason, it is neccessary to clear the associative arrays like this, instead of simply declaring an empty array each time.
    unset dasDatasets
    declare -Ag crabNames
    declare -Ag dasDatasets
    declare -ag dasMasses=() # Regular array of mass points to use as keys (-a)

    # Note that everytime this function is called, $dasDatasets, $crabNames, $dasMasses, and $dasYear are overwritten.
    # $dasMasses, $crabNames, and $dasDatasets (along with $dasFile in the next function) fill the sample files, 
    # and are used by the python script buildDict.py to build a python dictionary to help submit crab jobs.

    # Search DAS, store results:
    declare -a dasResults=( $(dasgoclient -query="dataset dataset=/${dasFront[$particle]}*${dasBack[$particle]}*/${dasYear}*-106X*/MINIAODSIM datatype=$datatype") )
    
    # Special case for RSGluon TT samples
    declare -a dasExtra=()
    if [[ $particle == "tt" ]]; then 
        dasExtra=( $(dasgoclient -query="dataset dataset=/RSGluon*${dasBack[$particle]}*/${dasYear}*-106X*/MINIAODSIM datatype=$datatype") )
        dasResults+=( "${dasExtra[@]}" )
    fi

    v2Counter=0 # Counter to keep track of how many datasets need a v2 still
    k=0 # Index for $dasMasses
    for dataset in ${dasResults[*]}; do # Fill arrays if datasets pass criteria:

        # The 2016 datasets will have the 2016_APV (2015) ones mixed in. Skip these datasets:
        if [[ $year == "2016" ]] && [[ $dataset =~ "RunIISummer20UL16MiniAODAPV" ]] ; then continue; fi

        # Skip QCD datasets with low pT, and also any MuEnriched datasets that may be present:
        if [[ $particle == "QCD" ]] && [[ $dataset =~ ("15to30"|"30to50"|"50to80"|"80to120"|"120to170"|"170to300"|"300to470"|"Enriched"|"bcToE") ]]; then continue; fi


        # Special case for RSGluon TT samples
        if [[ $dataset =~ "RSGluon" ]] && [[ $dataset =~ ("M-500_TuneCP5"|"M-1000_TuneCP5"|"M-1500_TuneCP5"|"M-2000_TuneCP5"|"M-2500_TuneCP5"|"M-3000_TuneCP5") ]]; then continue; fi
        if [[ $dataset =~ "RSGluon" ]]; then trimString=${dataset#*"RSGluonToTT_M-"}
        # Now trim the dataset string to get the mass point and build the crab directory name:
        else                                 trimString=${dataset#*${dasFront[$particle]}}; fi # Trims the corresponding $dasFront string from the front of the $dasResults string 
        massPoint=${trimString%${dasBack[$particle]}*} # Trims the rest of the back of $trimString; now $massPoint is the mass point of the dataset

        # Check for special cases: 
        if [[ $particle  == "HH"   ]] && [[ $massPoint == "60000" ]]; then massPoint="6000"; fi # 6000GeV Higgs dataset is mislabeled as 60000Gev
        if [[ $particle  == "QCD"  ]] && [[ $dataset   =~ "Flat"  ]]; then massPoint="Flat"; fi # The 15to7000 QCD pT sample is the flattened pT
        if [[ $year      == "2018" ]] && [[ $massPoint =~ "Flat"  ]]; then ((v2Counter++));  fi # There is no v1 2018 QCD Flat dataset, so v2Counter needs to be incremented manually

        # Check if current mass point is the same as previous mass point. If so, then the current dataset is a v2, and the previous dataset is the corresponding v1.
        if (( $k != 0 )) && [[ $massPoint == ${dasMasses[$k-1]} ]] ; then 
            ((v2Counter++)) # If dataset is a v2, increment $v2Counter. The code after the if statement will update the v1 entry to the v2 entry.
        else
            dasMasses[$k]=$massPoint # If the dataset is a v1, then add the new mass point to $dasMasses
            ((k++)) # Increment $dasMasses index
        fi

        dasDatasets[$massPoint]=$dataset # Store dataset

        if [[ $particle == "tt" ]]; then # Special case for top samples, which can have different datasets with different widths for the same mass 1000_W100
            if [[ $dataset =~ "RSGluon" ]]; then # Special case for RSGluon, which does not have a width
                crabNames[$massPoint]="RSGluonToTT_M_${massPoint}GeV_trees" 
            else
                massString=${massPoint%"_W"*} # Trims $massPoint from the back, leaving just the mass
                widthString=${massPoint#*"_W"} # Trims $massPoint from the front, leaving just the width
                crabNames[$massPoint]=${crabTemplate[$particle]/"MASSWIDTH"/"${massString}GeV_W_${widthString}GeV"} # Replace "MASSWIDTH" with current mass point/width, store crab directory name 
            fi
        else # All other particles do not have widths, can be treated the same:
            crabNames[$massPoint]=${crabTemplate[$particle]/"MASS"/${massPoint}} # Replace "MASS" with current mass point, store crab directory name 
        fi
    done 
    dasMasses=($(echo ${dasMasses[*]}| tr " " "\n" | sort -n)) # Sort $dasMasses
}

# This function checks DAS for files for each dataset. Since each dataset needs to check DAS individually, this part can take quite a while, especially if DAS is being slow.
# dasgoclient will return an array of file names, but we want each file to have about the same events. So we check DAS twice for each dataset, once for the names, once for the list of nevents.
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

#==================================================================================
# Fill Samples ////////////////////////////////////////////////////////////////////
#==================================================================================

# This loop will create the sample text files, search for the datasets on DAS, check for version, and fill the text files accordingly.
# It will also check for a previous set of sample files, and report any new datasets.
for datatype in ${myDatatypes[*]}; do # Loop over user chosen datatypes
    if [[ $datatype == "data" ]] ; then break; fi # skips the loop for data, not implemented yet
    echo "\n${PURP}Beginning $datatype...${NC}"

    for year in ${myYears[*]}; do # Loop over user chosen years
        echo "\n${GRN}Beginning $year...${NC}"

        # Define file names
        fileToWrite="../../samples/${datatype}_${year}.txt"
        filePrevious="../../samples/previous/${datatype}_${year}.txt"
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
            echo "${YEL}No current or previous sample files detected. Will build fresh ${GRN}$year ${PURP}$dat ${YEL}sample files from only the user inputted${NC} ${CYAN}particle ${YEL}selections. This might take a few minutes."
        fi

        echo "# Mass Point, Crab Directory, \tDataset Name, \t File With Most Events" >> $fileToWrite # Header for sample file
        for particle in ${allParticles[*]}; do # Loop over particles
            echo "\n${CYAN}Beginning $particle...${NC}"
            echo "#$particle" >> $fileToWrite # Labels the particle sections in sample file

            if [[ -f "$filePrevious" ]]; then # If a previous sample file exists, read in the corresponding particle section of data
                targetSection=false
                declare -a dasPrevious=()
                declare -a massPrevious=()
                declare -a crabPrevious=()
                declare -a dasFilePrevious=()
                file=$filePrevious
                OLDIFS=$IFS # Preserve the old IFS to reinstate it later
                IFS=',' # This lets us read in the comma separated sample files
                # This loop reads the previous sample file, stores values to arrays
                while read first second third fourth; do 
                    if [[ "$first" == "#$particle" ]]; then # Triggers when reaching the beginning of the relevant lines for this $particle loop
                        targetSection=true
                    elif [[ "$targetSection" == true ]]; then
                        if [[ ${allParticles[*]} =~ "${first:1}" ]]; then break; fi # Ends while loop when the next particle section of datasets begins 
                        massPrevious+=( "$first" )
                        crabPrevious+=( "$second" )
                        dasPrevious+=( "$third" )
                        dasFilePrevious+=( "$fourth" )
                    fi        
                done < "$filePrevious"
                IFS=$OLDIFS # Resets $IFS so the rest of the code works

                # Now compare with current datasets
                if [[ ${myParticles[*]} =~ $particle ]]; then # Checks if user specified this particle. If so, check DAS for updates to the list of datasets.
                    checkDASDatasets $particle $year $datatype # Check DAS for user chosen particle/year/datatype set, which loads several arrays/strings/counters.
                    j=0 # Index for the set of Previous arrays (if there is a new mass point found by checkDASDatasets, then the size of $dasDatasets and $dasPrevious will not match)
                    newCounter=0 # Keeps track of how many new files are found

                    for mass in ${dasMasses[*]}; do # Begin loop over dataset results
                        if [[ ${dasDatasets[$mass]} == ${dasPrevious[$j]} ]]; then # Copy case: If the datasets match, simply copy the old information. 
                            # Write the dataset to the samples file; using Previous results saves time checking DAS for individual files:
                            echo "${massPrevious[$j]},${crabPrevious[$j]},${dasPrevious[$j]},${dasFilePrevious[$j]}" >> $fileToWrite 
                            ((j++)) # Increments Previous array index
                        
                        elif [[ $mass == ${massPrevious[$j]} ]]; then # v2 case: If the datasets don't match, but the masses do, then there is a new v2 dataset.
                            echo "${YEL}NEW v2 DATASET DETECTED:${NC} ${dasDatasets[$mass]}" 
                            # Check DAS for file with most events, and write new dataset information to samples file:
                            checkDASFiles ${dasDatasets[$mass]} # Checks DAS for files associated with this dataset. Loads $dasFile and $dasEvent strings.
                            echo "$mass,${crabNames[$mass]},${dasDatasets[$mass]},${dasFile}" >> $fileToWrite 
                            ((j++)) # Increments Previous array index 

                        else # New file case: If the mass point doesn't match, it is a new mass point.
                            echo "${YEL}NEW MASS POINT DETECTED:${NC} ${dasDatasets[$mass]}"     
                            # Check DAS for file with most events, and write new dataset information to samples file:
                            checkDASFiles ${dasDatasets[$mass]} # Checks DAS for files associated with this dataset. Loads $dasFile and $dasEvent strings.
                            echo "$mass,${crabNames[$mass]},${dasDatasets[$mass]},${dasFile}" >> $fileToWrite 
                            ((newCounter++)) # Increment new files counter
                        fi
                    done

                    echo "\t${YEL}${v2Counter} ${BLUE}out of ${YEL}${#dasMasses[@]} ${BLUE}datasets were ${YEL}v2 datasets, ${BLUE}and ${YEL}$newCounter new mass points ${BLUE}were found.${NC}"
                    echo "\t${BLUE}Wrote${NC} /${CYAN}${dasFront[$particle]}${NC}[masses]${CYAN}${dasBack[$particle]}${NC}*/${GRN}${dasYear}${NC}*/${PURP}MINIAODSIM ${BLUE}datasets to:${NC} ${fileToWrite}"

                else # If the user did not specify the particle, and a Previous sample file exists, simply copy the previous file for that particle.
                    echo "${CYAN}Copying $particle...${NC}"        
                    for j in ${!dasPrevious[*]}; do
                        # Write the dataset to the samples file; using Previous results saves time checking DAS for individual files
                        echo "${massPrevious[$j]},${crabPrevious[$j]},${dasPrevious[$j]},${dasFilePrevious[$j]}" >> $fileToWrite 
                    done
                fi     
            else # Triggers if there is not a previous sample file
                if [[ ${myParticles[*]} =~ $particle ]]; then # Checks if user specified this particle. If so, check DAS for datasets and fill sample file.
                    checkDASDatasets $particle $year $datatype # Check DAS for user submitted particle/year/datatype set, loads several arrays/strings/counters, prints out v1 vs. v2 dataset counters
                    
                    for mass in ${dasMasses[*]}; do # Begin loop over dataset results
                        # Check DAS for file with most events, and write new dataset information to samples file:
                        checkDASFiles ${dasDatasets[$mass]} # Checks DAS for files associated with this dataset. Loads $dasFile and $dasEvent strings.
                        echo "$mass,${crabNames[$mass]},${dasDatasets[$mass]},${dasFile}" >> $fileToWrite 
                    done
                    
                    echo "\t${YEL}${v2Counter} ${BLUE}out of ${YEL}${#dasMasses[@]} ${BLUE}datasets were ${YEL}v2 datasets.${NC}"
                    echo "\t${BLUE}Wrote${NC} /${CYAN}${dasFront[$particle]}${NC}[masses]${CYAN}${dasBack[$particle]}${NC}*/${GRN}${dasYear}${NC}*/${PURP}MINIAODSIM ${BLUE}datasets to:${NC} ${fileToWrite}"
                else
                    echo "${CYAN}Skipping $particle...${NC}"
                fi
            fi
        done
        echo "\n${YEL}Checkout your new list of files at:${NC} $fileToWrite"
    done
done

# Build dictionary from sample files
echo
python buildDict.py

# Undo the alias used for this script
unalias echo