#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainBEST.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST: The Boosted Event Shape Tagger ////////////////////////
#==================================================================================

import time
startTime = time.time() # Tracks how long script takes

# modules
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import os

# get stuff from modules
from sklearn import preprocessing

# user modules
# import tools.functions as tools

# Plot BEST inputs before scaling (specific h5 file)
# Load all files, all vars.
#   Check for -999.99 and NaNs, etc. 
# Unscale BEST inputs, plot, compare
# Plot scaled BEST inputs to check 
# use a mask, and start with the 57 oldBEST vars
# reference the previous analysis, redo it exactly the same
# being certain of what the oldBEST issue was 
#   (# of vars per boost? Boosts? Arch? all need improvement, PROVE this) 
#  

#==================================================================================
# Load Data ///////////////////////////////////////////////////////////////////////
#==================================================================================
# print("Plotting pT", suffix)

# Load Mask
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask.txt"
maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_noDeepAK8noNJets_trimIso.txt"
maskFile = open(maskPath, "r")
varDict = {}
allVars = []
for line in maskFile:
    index, var = line.split(':')
    var = var.strip()
    varDict[index] = var
    allVars.append(var)
maskFile.close()
print(maskPath + " chosen; mask size " + str(len(allVars)))
myMask = [True if str(i) in varDict else False for i in range(596)]

sampleFileTypes = ["WW","ZZ","HH","TT","BB","QCD"]
sampleTypes     = ["W","Z","H","t","b","QCD"]
h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
splitSets = ["train","validation","test"]
for splitSet in splitSets:
    print("Beginning " + splitSet)
    print("Loading pre scale h5py files")
    preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + splitSet + "_flattened.h5","r")["BES_vars"])[:,myMask] for mySample in sampleFileTypes]

    print("Loading post scale h5py files")
    postScaleEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + splitSet + "_flattened_standardized.h5","r")["BES_vars"])[:,myMask] for mySample in sampleFileTypes]

    print("Pre scale events shape:", [arr.shape for arr in preScaleEvents])
    print("Post scale events shape:",[arr.shape for arr in postScaleEvents])

    print("Unscaling...")
    scalePath = "/uscms/home/bonillaj/nobackup/Brendan/CMSSW_10_2_18/src/centralBEST/BEST/training/ScalerParameters_" + splitSet + ".txt"
    scaleFile = open(scalePath, "r")
    means  = []
    scales = []
    i = -1
    for line in scaleFile:
        i += 1
        if str(i) not in varDict: continue
        mean, vari = line.split(',')
        vari = vari.strip()
        means.append(float(mean))
        # scales.append(float(vari))
        scales.append(float(vari) ** 2)
    scaleFile.close()

    scaler = preprocessing.StandardScaler()
    scaler.mean_ = np.array(means)
    scaler.scale_ = np.array(scales)
    # scaler.var_ = np.array(scales)
    unScaledEvents = [scaler.inverse_transform(scaledEvents) for scaledEvents in postScaleEvents]
    print("Unscaled events shape:",[arr.shape for arr in unScaledEvents])

    print("Plotting pT...")


    plotDir = "plots/BESvars/" + splitSet + "/"
    plt.figure()
    for index, var in enumerate(allVars):
        # print("Plotting " + var)
        saveDir = plotDir + var + "_" + splitSet + "/"
        if not os.path.isdir(saveDir): os.makedirs(saveDir)

        # --- Create Pre Scale histogram, legend and title ---
        title = var + "_" + splitSet + "_preScale" 
        for i, array in enumerate(preScaleEvents):
            plt.hist(array[:,index], bins=51, histtype='step')
            # Check for bad values
            minVal = np.amin(array[:,index])
            if minVal == -999.99: print("BAD VALUE: ", title, sampleTypes[i])
        plt.legend(frameon=False, labels = sampleTypes)
        plt.title( title )
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.clf()
        
        # --- Create Post Scale histogram, legend and title ---
        title = var + "_" + splitSet + "_postScale" 
        for i, array in enumerate(postScaleEvents):
            plt.hist(array[:,index], bins=51, histtype='step')
            # Check for bad values
            minVal = np.amin(array[:,index])
            if minVal == -999.99: print("BAD VALUE: ", title, sampleTypes[i])
        plt.legend(frameon=False, labels = sampleTypes)
        plt.title( title )
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.clf()

        # --- Create Unscaled histogram, legend and title ---
        title = var + "_" + splitSet + "_unscaled" 
        for i, array in enumerate(unScaledEvents):
            plt.hist(array[:,index], bins=51, histtype='step')
            # Check for bad values
            minVal = np.amin(array[:,index])
            if minVal == -999.99: print("BAD VALUE: ", title, sampleTypes[i])
        plt.legend(frameon=False, labels = sampleTypes)
        plt.title( title )
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.clf()

    plt.close()


# Check how long the script took to run
timelog = open("logs/timelog_plotBESTInputs", "a") 
timeTaken = divmod(time.time() - startTime, 60.)
timeMessage = "Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
print(timeMessage)
timelog.write(timeMessage)
timelog.close