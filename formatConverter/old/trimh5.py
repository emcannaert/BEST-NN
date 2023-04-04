#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trimh5.py ///////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Samantha Abbott ------------------------------------------------------
# This script trims some bad events from the BB and QCD h5 files.  ////////////////
# Bad variable is "284:ak8SoftDropFrame_jet_energy0" //////////////////////////////
# The script confirms that the events were removed correctly. /////////////////////
# SoftDropFrame still being tested, could be left out of final release. /////////// 
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Figure out if we should keep this script in the final release.
# If we keep it, it needs to be more general and needs more comments.

import numpy as np
import h5py

# setTypes = ["test", "validation"]
setTypes = ["test", "validation", "train"]
# setTypes = ["test"]
sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
outFile = open("badEvents.txt", "w")
print("Identifying Bad Events...")
eventDict = {}
for mySet in setTypes:
    eventDict[mySet] = {}

    print("\n" + mySet)
    outFile.write(mySet + ":\n")
    for mySample in sampleTypes:
        eventDict[mySet][mySample] = []
        print("\n" + mySample)

        array  = np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_"+mySet+"_flattened.h5","r")["BES_vars"])[()]
        print("h5 file loaded; shape:", array.shape)
        outFile.write(mySample + "; shape: " + str(array.shape) + ":\n")

        eventInd, varInd = np.where(array == -999.99)
        if len(eventInd) == 0: continue

        tempDict = {}
        for i, evIndex in enumerate(eventInd):
            vrIndex = varInd[i]
            if evIndex not in tempDict.keys(): 
                tempDict[evIndex] = [vrIndex]
                eventDict[mySet][mySample].append(evIndex)
            else:
                tempDict[evIndex].append(vrIndex)

        for event, vars in tempDict.items():
            outFile.write(str(array[event,vars[0]]) + ", event " + str(event) + ", " + str(len(vars)) + " bad vars: " + str(vars) + "\n" )            
            # print("event: " +str(event) + ", " + str(len(vars)) + " total bad vars \nbad vars: " + str(vars) )
            print("event: " +str(event) + ", " + str(len(vars)) + " total bad vars")
        outFile.write("\n")
    outFile.write("\n")


print("\nBad events identified:\n")
outFile.write("\nList of bad events:\n")
for mySet, sampleDict in eventDict.items():
    print("\n" + mySet)
    outFile.write(str(mySet) + ":\n")
    for mySample, badEvents in sampleDict.items():
        if len(badEvents) == 0: continue
        print(mySample, badEvents)
        outFile.write(str(mySample) + ": " + str(badEvents) + "\n" )
# outFile.close()
print("Checkout badEvents.txt for a summary")


print("\nTrimming...")
for mySet, sampleDict in eventDict.items():
    print("\n" + mySet)

    for mySample, badEvents in sampleDict.items():
        if len(badEvents) == 0: continue
        print("\n" + mySample)
        f  = h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_"+mySet+"_flattened.h5","r+")
        # f  = h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_"+mySet+"_flattened.h5","r")
        dset = f['BES_vars']

        print("Extracting data to array...")
        oldDS = dset[()]
        oldShape = oldDS.shape
        print("Array shape before trimming: " + str(oldShape) )

        print("There are " + str(len(badEvents)) + " Bad Events: \n" + str(badEvents))
    
        print("Creating trimmed array...")
        newDS = np.delete(oldDS, badEvents, 0)
        del oldDS
        newShape = newDS.shape
        print("Array shape after trimming: " + str(newShape) )

        print("Verifying array was trimmed correctly...")
        if not ( oldShape[0] - newShape[0] == len(badEvents) ):
            print("ERROR, events don't match")
            quit()
        elif not ( oldShape[1] == newShape[1] ) :
            print("ERROR, variables don't match")
            quit()
        elif np.isnan(newDS[...,284]).any() or np.isposinf(newDS[...,284]).any() or (len( np.where(newDS == -999.99)[0] ) != 0):
            print("ERROR, NaN or +Inf or -999.99 values remain in new dataset")
            print("Nan:")
            print(np.where(np.isnan(newDS[...,284])))
            print("+Inf:")
            print(np.where(np.isposinf(newDS[...,284])))
            print("-999.99:")
            print(np.where(newDS == -999.99))
            quit()
        else:
            print("Array trimmed correctly! \nUpdating dataset in h5 file...")

        print("Dataset shape before update: " + str(dset.shape))
        dset.resize(newShape[0], axis=0)
        print("Dataset shape after resize: " + str(dset.shape))
        dset[:] = newDS
        del newDS
        print("Dataset shape after replacement: " + str(dset.shape))

        print("Verifying that dataset was updated correctly...")
        if not ( dset.shape == newShape ) :
            print("ERROR, shape is incorrect")
            quit()
        elif np.isnan(dset[...,284]).any() or np.isposinf(dset[...,284]).any() or (len( np.where(dset == -999.99)[0] ) != 0):
            print("ERROR, NaN or +Inf values remain in new dataset")
            print("Nan:")
            print(np.where(np.isnan(dset[...,284])))
            print("+Inf:")
            print(np.where(np.isposinf(dset[...,284])))
            print("-999.99:")
            print(np.where(dset == -999.99))
            quit()
        else:
            print("Dataset updated correctly!!!")

print("\nUpdated all datasets!")

print("Recording new shapes:\n")
outFile.write("\nNew shapes:\n")
for mySet, sampleDict in eventDict.items():
    outFile.write(str(mySet) + ":\n")

    for mySample, badEvents in sampleDict.items():
        f  = h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_"+mySet+"_flattened.h5","r")
        dset = f['BES_vars']
        outFile.write(str(mySample) + " new shape: " + str(dset.shape) + "\n" )

outFile.close()
print("\n\n\n\nHI JOHAN TELL HOLLIS WE SAID HI OK BYE")