#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# merge.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Mark Samuel Abbott ---------------------------------------------------
# This script merges the RSGluon h5 files into the main Z h5 files. ///////////////
# The script confirms that the merge was successful.  /////////////////////////////
# Only needed this script since we added the RSGluon data later. ////////////////// 
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Figure out if we should keep this script in the final release.
# If we keep it, it needs to be more general and needs more comments.


# modules
import numpy as np
import h5py

setTypes = ["train", "validation", "test"]
files = { "ZPrime":{}, "RSGluon":{} }
verifyDict = { "ZPrimeBefore":{ "shape":{}, "chunk":{} }, "ZPrimeAfter":{ "shape":{}, "chunk":{} }, "RSGluon":{ "shape":{}, "chunk":{} }, }
for setType in setTypes:
    # Open file, grab keys, and NEvents
    print("\nBeginning " + setType)
    ZPrime = h5py.File("/uscms/home/bonillaj/nobackup/h5samples_ULv1/TTSample_2017_BESTinputs_"+setType+".h5","a")
    RSGluon = h5py.File("/uscms/home/msabbott/nobackup/abbott/CMSSW_10_6_27/src/BEST/formatConverter/h5samples/RSGSample_2017_BESTinputs_"+setType+".h5", 'r')
    files["ZPrime"][setType]  = ZPrime
    files["RSGluon"][setType] = RSGluon

    dataKeys = list(ZPrime.keys())
    print(dataKeys)
    print(RSGluon[dataKeys[0]].shape)
    totalEvents = RSGluon[dataKeys[0]].shape[0]

    # Store shape and chunk info to verify script ran correctly
    for key in verifyDict.keys():
        verifyDict[key][setType] = {}
        verifyDict[key]["shape"][setType] = {}
        verifyDict[key]["chunk"][setType] = {}
        
    for myKey in dataKeys:
        verifyDict["ZPrimeBefore"][setType][myKey] = { "shape":ZPrime[myKey].shape, "chunk":ZPrime[myKey].chunks }
        verifyDict["RSGluon"][setType][myKey] = {"shape":RSGluon[myKey].shape, "chunk":RSGluon[myKey].chunks}


    counter = 0
    while (counter < totalEvents):
        batchSize = 500000 if (totalEvents > counter+500000) else (totalEvents-counter)
        print("Batch size of ",batchSize,", at counter ",counter)
        for myKey in dataKeys:
            print("MyKey",myKey)
            dsetNP = np.array(RSGluon[myKey][counter:counter+batchSize])

            # append the dataset
            ZPrime[myKey].resize(ZPrime[myKey].shape[0] + len(dsetNP), axis=0)
            ZPrime[myKey][-len(dsetNP) :] = dsetNP
        counter += batchSize
    
    for myKey in dataKeys:
        verifyDict["ZPrimeAfter"][setType][myKey] = { "shape":ZPrime[myKey].shape, "chunk":ZPrime[myKey].chunks }

print("Done with appending. Verifying script ran correctly...\n")

failFlag = False
for setType in setTypes:
    print("\nVerifying " + setType)
    for myKey in dataKeys:
        print("Starting " + myKey)
        ZShapeBefore = verifyDict["ZPrimeBefore"][setType][myKey]["shape"]
        ZShapeAfter  = verifyDict["ZPrimeAfter"][setType][myKey]["shape"]
        GluonShape = verifyDict["RSGluon"][setType][myKey]["shape"]
        ZChunkBefore = verifyDict["ZPrimeBefore"][setType][myKey]["chunk"]
        ZChunkAfter  = verifyDict["ZPrimeAfter"][setType][myKey]["chunk"]
        GluonChunk = verifyDict["RSGluon"][setType][myKey]["chunk"]
        print("Shape TT_before, RSG_added, TT_after: ", ZShapeBefore, GluonShape, ZShapeAfter)
        print("Chunks TT_before, RSG_added, TT_after: ", ZChunkBefore, GluonChunk, ZChunkAfter)

        if ZChunkBefore == ZChunkAfter:
            print("Chunks Match!")
        else:
            print("ERROR: CHUNKS DO NOT MATCH")
            failFlag = True

        if ZShapeBefore[0] + GluonShape[0] == ZShapeAfter[0]:
            print("Events Match!")
        else:
            print("ERROR: EVENTS DO NOT MATCH")
            failFlag = True

        if myKey == "BES_vars": 
            if ZShapeBefore[1] == ZShapeAfter[1]:
                print("Variables Match!")
            else:
                print("ERROR: VARIABLES DO NOT MATCH")
                failFlag = True
        else:
            if (ZShapeBefore[1] == ZShapeAfter[1]) and (ZShapeBefore[2] == ZShapeAfter[2]):
                print("Variables and Candidates Match!")
            else:
                print("ERROR: VARIABLES AND/OR CANDIDATES DO NOT MATCH")
                failFlag = True

if failFlag:
    print("Something doesn't match!!!! Check output above for ERROR")
else:
    print("Appending completed successfully!!!")
