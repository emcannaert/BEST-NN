#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# sampleSplitter.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Johan S Bonilla, Brendan Regnery -------------------------------------
# This program splits h5f files into 3 smaller but equal orthogonal files       ///
# The point is to create separate train/validation/test samples of equal size   ///
# Inputs should be h5f files (can be flatted, or staright from formatConverter) ///
# Output should be three sets of hd5f files: training, validation, testing      ///
#----------------------------------------------------------------------------------

# fuss with defaults? make them uniform across convert,split,flatten?

# modules
import numpy as np
import h5py
import argparse
import os
import time
from sklearn.model_selection import train_test_split

# Global variables
# listOfSamples = ["BB","HH","QCD","TT","WW","ZZ"]
listOfSamples = ["Zt","Ht","Wb","QCD","TTBar"]
listOfYears = ["2018"]

# Helper functions
def splitFileSKL(inputPath, outDir, debug, userBatchSize):
    print("Starting clock")
    startTime = time.time()
    
    setTypes = ["train", "validation", "test"]

    # Open file, grab keys, and NEvents
    inputFile = h5py.File(inputPath,"r")
    dataKeys = list(inputFile.keys())
    print(dataKeys)
    print(inputFile[dataKeys[0]].shape)
    totalEvents = inputFile[dataKeys[0]].shape[0]
    print("input/output paths are:",inputPath, "/",outDir)
    # Create data frame and output files to handle copied information
    # Make h5f file to store the images and BES variables
    outName = outDir+inputPath.split('.')[-2].split('/')[-1]
    h5fTrain = h5py.File(outName+"_train.h5","w")
    h5fValidation = h5py.File(outName+"_validation.h5","w")
    h5fTest = h5py.File(outName+"_test.h5","w")
        
    besData = {}
    for setType in setTypes:
        besData[setType] = {}

    counter = 0
    while (counter < totalEvents):
        batchTime = time.time()
        batchSize = userBatchSize if (totalEvents > counter+userBatchSize) else (totalEvents-counter)
        print("Batch size of ",batchSize,", at counter ",counter)
        print("Starting key loop")
        for myKey in dataKeys:
            print("MyKey",myKey)
            keyTime = time.time()
            # dsetH5 = inputFile[myKey]
            dsetShape  = inputFile[myKey].shape
            dsetChunks = inputFile[myKey].chunks 
            dsetNP     = np.array(inputFile[myKey][counter:counter+batchSize])
            print("NPdset creation time:", time.time()-keyTime)
            ## Shuffle=True shuffles the incoming data set.
            ## Random state sets the seed. The values are meaningless, but the same value leads to same results
            ## The most important thing here is that each key is passed the same random state so the same events are split and kept

            # output1 = train_test_split(dsetNP, train_size=0.8, shuffle=True, random_state=42) # Full dataset -> 80% train (output1[0]), 20% 'test' (output1[1]) -> next line
            # output2 = train_test_split(output1[1], train_size=0.5, shuffle=True, random_state=24) # 20% 'test' -> 10% validation (output2[0]), 10% test (output2[1])

            train, temp = train_test_split(dsetNP, train_size=0.8, shuffle=True, random_state=42) # Full dataset -> 80% train (train), 20% 'test' (temp) -> next line
            valid, test = train_test_split(temp, train_size=0.5, shuffle=True, random_state=24) # 20% 'test' -> 10% validation (valid), 10% test (test)
            print("Outdset creation time:", time.time()-keyTime)
            if counter == 0:
                if myKey == "BES_vars": # max shape by # of vars
                    besData["train"][myKey] = h5fTrain.create_dataset(myKey, data=train, maxshape=(None, dsetShape[1]), chunks = (dsetChunks[0], dsetChunks[1]), compression='lzf', shuffle=True)
                    besData["validation"][myKey] = h5fValidation.create_dataset(myKey, data=valid, maxshape=(None, dsetShape[1]), chunks = (dsetChunks[0], dsetChunks[1]), compression='lzf', shuffle=True)
                    besData["test"][myKey] = h5fTest.create_dataset(myKey, data=test, maxshape=(None, dsetShape[1]), chunks = (dsetChunks[0], dsetChunks[1]), compression='lzf', shuffle=True)
                    print("DS store time:", time.time()-keyTime)
                else: # max shape by # of pfcands (or SV's) and # of vars
                    besData["train"][myKey] = h5fTrain.create_dataset(myKey, data=train, maxshape=(None, dsetShape[1], dsetShape[2]), chunks = (dsetChunks[0], dsetChunks[1], dsetChunks[2]), compression='lzf', shuffle=True)
                    besData["validation"][myKey] = h5fValidation.create_dataset(myKey, data=valid, maxshape=(None, dsetShape[1], dsetShape[2]), chunks = (dsetChunks[0], dsetChunks[1], dsetChunks[2]), compression='lzf', shuffle=True)
                    besData["test"][myKey] = h5fTest.create_dataset(myKey, data=test, maxshape=(None, dsetShape[1], dsetShape[2]), chunks = (dsetChunks[0], dsetChunks[1], dsetChunks[2]), compression='lzf', shuffle=True)
                    print("DS store time:", time.time()-keyTime)
            else:
                # append the dataset
                besData["train"][myKey].resize(besData["train"][myKey].shape[0] + len(train), axis=0)
                besData["train"][myKey][-len(train) :] = train
                besData["validation"][myKey].resize(besData["validation"][myKey].shape[0] + len(valid), axis=0)
                besData["validation"][myKey][-len(valid) :] = valid
                besData["test"][myKey].resize(besData["test"][myKey].shape[0] + len(test), axis=0)
                besData["test"][myKey][-len(test) :] = test 
                print("DS store time:", time.time()-keyTime)
            print("Key iteration time:", time.time()-keyTime)
            keyTime = time.time()
        print("Batch time:", time.time()-batchTime)
        counter += batchSize
    print("Splitting time:", time.time()-startTime)

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) WW,ZZ,BB',
                        required=True)
    parser.add_argument('-y', '--years',
                        dest='years',
                        help='<Required> Which (comma separated) years to process. Examples: 1) --all; 2) 2016,2018',
                        required=True)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="~/nobackup/h5Dir/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default='~/nobackup/h5Dir/')
    parser.add_argument('-bs', '--batchSize',
                        type=int,
                        required=True)
    parser.add_argument('-d','--debug',
                        action='store_true')
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if not args.years == "all": listOfYears = args.years.split(',')
    if args.debug:
        print("Samples to process: ", listOfSamples)
        print("Years to process: ", listOfYears)

    # Check existance of directories you need
    if not os.path.isdir(args.h5Dir): 
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isdir(args.outDir):
        os.mkdir(args.outDir)

    for year in listOfYears:
        for sampleType in listOfSamples:
            print("Processing", sampleType)
            inputPath = args.h5Dir+"/"+sampleType+"Sample_"+year+"_BESTinputs.h5"
            splitFileSKL(inputPath, args.outDir, args.debug, args.batchSize)
        
            
    print("Done")

