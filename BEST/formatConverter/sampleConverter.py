	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# sampleConverter.py //////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Mark Samuel Abbott, Johan S Bonilla, Brendan Regnery -----------------
# This program converts root ntuples to the python format necessary for training //
# Inputs should be root files from preprocess
# Output should be three sets of hd5f files: trainingSet, validationSet, testignSet
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Make firstConvert its own function
# Add more comments

import time

startTime = time.time() # Tracks how long script takes

# modules
import ROOT as root
import uproot
import numpy as np
import h5py
import argparse
import os


# Enter batch mode in root, so python can access displays
root.gROOT.SetBatch(True)

# Global variables
listBESvars = False
stopAt = None
listOfSamples = ["Zt","Ht","Wb", "QCD","TTBar"]
# listOfSamples = ["HH"]
#listOfSamples = ["RSG"]
years = ["2018"]
treeName = "run/jetTree"

# Each of these key lists represent a different type of h5py dataset:
besKeys = []  # Standard BES variables, only 1 value per event(jet)
depKeys = []  # PF variables that are frame dependent, several values per event(jet)
invKeys = []  # PF variables that are frame invariant, several values per event(jet)
#svKeys  = []  # SV variables, several values per event(jet). Does not include 'nSecondaryVertices'

# Labels for PF variables (both frame dependent and invariant)
pfFrameLabels   = [] # Labels for each Frame
depPFInfoLabels = [] # Labels for frame dependent variables
invPFInfoLabels = [] # Labels for frame invariant variables

# Flag for code that only needs to happen once
firstConvert = True 

#==================================================================================
# Convert /////////////////////////////////////////////////////////////////////////
#==================================================================================
def convert(eosDir, outDir, sampleType, year, debug):
    # Find file paths that are relevant in eos and write the paths to a txt file
    # At the moment the user is expected to make the txt file by eosls dir >> listOf<sampleType>filePaths<year>.txt
    # If the files are on eos, their paths should have 'root://cmseos.fnal.gov//eosDir' as a prefix
    # In the future this should be automated, hard part is working with eos from python...
    # This file should (for now) live in your current directory (BEST/formatConverter/eosSamples/listOf<sampleType>filePath<year>.txt)

    # Open file and read lines (individual file paths)
    if debug: print("Reading from", eosDir+"listOf"+sampleType+"FilePaths"+year+".txt")
    with open(eosDir+"/listOf"+sampleType+"FilePaths"+year+".txt", 'r') as myFile:
        # Read file paths from txt file
        fileList = myFile.read().splitlines()
        if debug: print (fileList)
        
        # Make h5f output file to store the images and BES variables
        h5fPath = outDir+sampleType+"Sample_"+year+"_BESTinputs.h5"
        if debug: print ("Writing h5f file to",h5fPath)
        h5f = h5py.File(h5fPath,"w")

        numIter = 0
        besDS = None
        pfsvDS  = {}
        batchSize = 1000
	batchPrint = 10
        for arrays in uproot.iterate(fileList, treeName, entrysteps = batchSize, namedecode='utf-8'):
            
            # Load keys and labels
            # To make this code neater, I should change the first convert stuff to a function that is run once
            global firstConvert # Necessary for if statement to see firstConvert
            if firstConvert: # Variables are identical across sampleType and Year, so this code only needs to happen once:
                keys = arrays.keys()
                keys.sort()
                for key in keys :
                    if "PF_candidate" in key:
                        # Store PFcand info: XFrame_PF_candidate_Yinfo

                        myFrameLabel = key.split("_")[0]  # This gives 'XFrame'
                        if not myFrameLabel in pfFrameLabels: pfFrameLabels.append(myFrameLabel)
                        myInfoLabel  = key.split("_")[-1] # This gives 'Yinfo'
                        
                        if myFrameLabel == "AllFrame":

                            if myInfoLabel == "Weights": myInfoLabel = u'PUPPI_Weights' # Remove this the next BEST compile
                            if myInfoLabel == "PUPPIweights": myInfoLabel = u'PUPPI_Weights' # Remove this the next BEST compile

                            invPFInfoLabels.append(myInfoLabel) # Only one of each of these variables, so no if statement needed
                            invKeys.append(key)
                        else:
                            if not myInfoLabel  in depPFInfoLabels: depPFInfoLabels.append(myInfoLabel)
                            depKeys.append(key) 

                    elif "SV" in key:
			continue
                        #svKeys.append(key)
                    elif 'Frame_jet' in key: # Reclustered boosted jet energy, px, py, pz. Take four leading jets by energy (converting a list to four single variables to work with BES h5py dataset type)
                        if "Lab" in key: continue # Remove this after next BESTProducer.cc compile
                        besKeys.append(key+'0')
                        besKeys.append(key+'1')
                        besKeys.append(key+'2')
                        besKeys.append(key+'3')
                    else:
                        besKeys.append(key)

                if listBESvars == True:
                    print("There will be ", len(besKeys), " Input features stored")
                    print("There will be ", len(depKeys), " PF Frame Dependent features stored")
                    print("There will be ", len(invKeys), " PF Frame Invariant features stored")
                    #print("There will be ", len(svKeys),  " SV features stored")                    
                if listBESvars == True:
                    print("Here are the stored BES vars ", besKeys)
                    print("Here are the stored PFlow Frame Dependent cand vars ", depKeys)
                    print("Here are the stored PFlow Frame Invariant cand vars ", invKeys)
                    #print("Here are the stored Secondary Vertex vars ", svKeys)
                else:
                    print("If you would like to list the BES vars, set listBESvars = True at the beginning of the code")
                
                if debug:
                    print("pfFrameLabels: ",   pfFrameLabels)
                    print("depPFInfoLabels: ", depPFInfoLabels)
                    print("invPFInfoLabels: ", invPFInfoLabels)

                #pfFrameLabels.append("SV") # Not really a frame, but doing this allows us to handle SV vars with already existing code later on

                # Save legend for each dataset type so we know which matrix entry corresponds to which variable:
                savedVarsDict = {"BES":besKeys, "pfDEP":depPFInfoLabels, "pfINV":invPFInfoLabels}
                for varType, savedVars in savedVarsDict.items(): 
                    varFile = outDir+varType+"varList.txt"
                    if os.path.exists(varFile): os.remove(varFile)
                    with open(varFile,"w") as myBESlistFile:
                        for i, var in enumerate(savedVars):
                            myBESlistFile.write(str(i)+":"+var+"\n")
                
                print("Prints every " + str(batchPrint) + " batches, which is every " + str( batchSize*batchPrint ) + " events." )    
                # End of firstConvert if statement. From now on, we can skip this block of code
                firstConvert = False 
 

            thisBatchSize = min(batchSize,len(arrays[arrays.keys()[0]]))
            if ( (numIter % batchPrint) == 0 ): print(sampleType+": This batch (from, to, size):",numIter*thisBatchSize,(numIter+1)*thisBatchSize, thisBatchSize) # Display every 10,000 events
            (besDS, pfsvDS) = storeBESTinputs(h5f, numIter, arrays, besDS, pfsvDS, thisBatchSize, debug)
            
            # increment
            numIter += 1
            
            # if the stop iteration option is enabled
            if stopAt != None and stopAt <= besDS.shape[0] : 
                print("This program was told to stop early, please set 'stopAtIter = None' if you want it to run through all files")
                break


    print("Finished loading MC")
    return 

#==================================================================================
# Store BEST Inputs ///////////////////////////////////////////////////////////////
#==================================================================================
def storeBESTinputs(h5f, numIter, arrays, besDS, pfsvDS, thisBatchSize, debug):
    jetDF = {} # Make a data frame to store the BES variables and PF Candidates

    # Store BES variables
    besList = []
    for besKey in besKeys :
        if 'Frame_jet' in besKey:
            newArr1 = arrays[besKey[:-1]]
            nA2 = []

            for val in newArr1:
                number = int(besKey[-1])
                if len(val)>number: nA2.append(val[number])
                else:               nA2.append(0.0)

            newArr2 = np.array(nA2)
            if debug: print(besKey, newArr2.shape)
            besList.append(newArr2)

        else:
            if debug: print(besKey, arrays[besKey])
            besList.append(arrays[besKey])

    jetDF['BES_vars'] = np.array(besList).T

    if debug: print(jetDF['BES_vars'].shape)

    # Create dataset if first batch, otherwise append to dataset
    if numIter == 0: 
        besDS = h5f.create_dataset('BES_vars', data=jetDF['BES_vars'], maxshape=(None, len(besKeys)))
                                  #chunks = (10, len(besKeys)), compression="lzf", shuffle=True)
    else: 
        besDS.resize(besDS.shape[0] + len(jetDF['BES_vars']), axis=0)
        besDS[-len(jetDF['BES_vars']) :] = jetDF['BES_vars']

    if debug: print("Done with BESvars info")


    # Store PFcand and SV info: 
    maxPFlowCands = 50 
    maxSecVerts = 10 
    for myFrameLabel in pfFrameLabels:
        if debug: print("Copying "+myFrameLabel+" info")
        
        if myFrameLabel == "SV":
            myDFKey = 'SV_vars'
            vars = svKeys
            maxObjects = maxSecVerts
            chunkEvents = 50 # If batchSize = 1000, then one batch is 20 chunks
        else:
            myDFKey = 'PF_cands_'+myFrameLabel
            vars = invPFInfoLabels if (myFrameLabel == 'AllFrame') else (depPFInfoLabels)
            maxObjects = maxPFlowCands
            chunkEvents = 10 # If batchSize = 1000, then one batch is 100 chunks

        jetDF[myDFKey] = np.zeros((thisBatchSize,maxObjects,len(vars)), dtype=float)
                
        for varIndex, var in enumerate(vars):
            varKey = var if (myFrameLabel == "SV") else (myFrameLabel+"_PF_candidate_"+var)
            for i in range(0,thisBatchSize): # iterate over events
                a = arrays[varKey][i] # Variable value for one event (jet). Will be list of lists (list of PF candidates and list of values for each PFCand)
                for j in range(0,maxObjects): # while loop?
                    if j<len(a):
                        jetDF[myDFKey][i][j,varIndex] = a[j]
        # Create dataset if first batch, otherwise append to dataset
        if numIter == 0:
            pfsvDS[myDFKey] = h5f.create_dataset(myDFKey, data=jetDF[myDFKey], maxshape=(None, jetDF[myDFKey].shape[1],jetDF[myDFKey].shape[2])) 
                                                # chunks = (chunkEvents, maxObjects, len(vars)), compression="lzf", shuffle=True)
        else:
            pfsvDS[myDFKey].resize(pfsvDS[myDFKey].shape[0] + jetDF[myDFKey].shape[0], axis=0)
            pfsvDS[myDFKey][-len(jetDF[myDFKey]) :] = jetDF[myDFKey] 

    if debug: print("Done with PFcand info")

    if debug: print("Converted jets: ", besDS.shape[0] - len(jetDF['BES_vars']), " to ", besDS.shape[0])

    if debug: print("Finished storing BEST inputs")
    return (besDS, pfsvDS)


## Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments and set driving variables
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) all; 2) W,Z,b',
                        required=True)
    parser.add_argument('-y', '--years',
                        dest='years',
                        help='<Required> Which (comma separated) years to process. Examples: 1) all; 2) 2016,2017',
                        required=True)
    parser.add_argument('-sa', '--stopAt',
                        type=int,
                        default=-1)
    parser.add_argument('-eos','--eosDir',
                        dest='eosDir',
                        default="eosSamples/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="h5samples/")
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if not args.years == "all": years = args.years.split(',')
    if args.stopAt > 0: stopAt = args.stopAt

    # Diagnostic debug
    if args.debug:
        print("Samples to process:", listOfSamples)
        print("Years to process:", years)
        print("Reading every nEvents:", stopAt)

    # Make directories you need
    if not os.path.isdir(args.outDir): os.mkdir(args.outDir)

    # Loop over samples and convert each separately
    for sampleType in listOfSamples:
        for year in years:
            print("Processing", sampleType, year)
            convert(args.eosDir, args.outDir, sampleType, year, args.debug)
    
    print("Done")

    # Check how long the script took to run
    runf = open("timelog_converter", "a") 
    timeTaken = divmod(time.time() - startTime, 60.)
    runf.write("Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n")
    runf.close
