#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# scaleTest.py ////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Samantha Abbott //////////////////////////////////////////////////////
# This program tests the Standardized Inputs //////////////////////////////////////
#----------------------------------------------------------------------------------

import time
startTime = time.time() # Tracks how long script takes
"""
################################## NOTES TO SELF ##################################
# Add more comments, improve explanation at the top.
# Save model using joblib instead of saving the mean/variance.
# Make consistent with other scripts.
# Figure out what the issue with scaling is
#       Test by scaling and unscaling in the same script, then plotting. 

import numpy as np
import h5py
from sklearn import preprocessing
import tools.abbottFunctions as tools
import argparse, os
# from joblib import dump, load
from sklearn.externals.joblib import dump, load
from sklearn.compose import ColumnTransformer
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt

sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
# sampleTypes = ["WW"]
# sampleTypes = ["BB","HH","TT","WW","ZZ","QCD"]
# sampleTypes     = ["W","Z","H","t","b","QCD"]
# frameTypes     = ["Bottom","Higgs","Top","W","Z"]
# setTypes = ["validation", "test", "train"]
# setTypes = ["validation", "test"]
year = "2017"
suffix = "flattened"
h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
if not os.path.isdir(h5Dir): print(h5Dir, "does not exist")
"""

from re import M
from xml.dom.minicompat import NodeList


# modules
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import tensorflow as tf
import math

# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" # must set backend before importing keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.externals.joblib import load

# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
k.tensorflow_backend.set_session(tf.Session(config=config))

# user modules
# import tools.functions as tools
import tools.abbottFunctions as tools
from plotBESTPerformance import plotAll
from oldBEST import TrnValEvents, trainOldBEST

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
frameTypes = ["W","Z","Higgs","Top","Bottom"]

#try different batchsizes
BatchSize = 1200
# TrnValEvents = [ None, None]
# TrnValEvents = [ 2000000, 200000]
# TrnValEvents = [ 500000, 50000]
# TrnValEvents = [ 250000, 25000]
# TrnValEvents = [ 125000, 12500]
# TrnValEvents = [ 50000, 10000]

# trainMaxEvents      = TrnValEvents[0]
# validationMaxEvents = TrnValEvents[1]

setTypes = ["validation", "train"]

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

def plotBESVars(data, myLabels, scaleType, plotDir, thisSet, allVars, plotlist):
    print("Plotting " + scaleType + " " + thisSet)
    plt.figure()
    j = 0
    # for index, var in enumerate(allVars):
    for index in plotlist:
        # saveDir = plotDir + var + "_" + thisSet + "/"
        # if mask[index]: saveDir = plotDir + var + "/"
        # if mask[index]: saveDir = plotDir + var + "_" + thisSet + "/"
        # else:           continue  
        # print(saveDir)
        # else:             saveDir = plotDir + "/extraVars/" + var + "_" + thisSet + "/"

        var = varDict[str(index)]
        saveDir = plotDir + var + "/"
        if not os.path.isdir(saveDir): os.makedirs(saveDir)

        # --- Create histogram, legend and title ---
        title = var + "_" + scaleType
        # plt.hist(data[:,index], bins=51, histtype='step')
        # title = var + "_" + thisSet + "_" + scaleType 
        for i, array in enumerate(data):
            plt.hist(array[:,index], bins=51, histtype='step')
            # plt.hist(array[:,j], bins=51, histtype='step')
            # Check for bad values
            # minVal = np.amin(array[:,index])
            # if minVal == -999.99: print("BAD VALUE: ", title, myLabels[i])
        plt.legend(frameon=False, labels = myLabels)
        plt.title( title )
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.clf()
        j += 1
    plt.close()

def checkRepeat(j, keys):
    repeatList = []
    while j in keys:
        repeatList.append(j)
        j += 1
    # print(repeatList)
    return repeatList

if __name__ == "__main__":

    #==================================================================================
    # Load Data ///////////////////////////////////////////////////////////////////////
    #==================================================================================
    # print("Plotting pT", suffix)

    # Load Mask
    # maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask.txt"
    # # maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_noDeepAK8noNJets_trimIso.txt"
    # maskFile = open(maskPath, "r")
    # varDict = {}
    # # allVars = []
    # for line in maskFile:
    #     index, var = line.split(':')
    #     var = var.strip()
    #     varDict[index] = var
    #     # allVars.append(var)
    # maskFile.close()
    # print(maskPath + " chosen; mask size " + str(len(varDict.keys())))
    # # mask = [True if str(i) in varDict else False for i in range(596)]
    modelPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_compareFrames_BESonlyArch/All100Basic_allFrames/BEST_model_All100Basic_allFrames.h5"
    model = load_model(modelPath)
    print(model)
    print(model.summary() )
    print(model.history())

    quit()
    """    
    # maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask_allFrames.txt"
    maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_allFrames.txt"
    # maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_noDeepAK8noNJets_trimIso.txt"
    # maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/allBESTVars.txt"
    maskFile = open(maskPath, "r")
    allVars = []
    varDict = {}
    revDict = {}
    # dictMinMax = {}
    # dictMaxAbs = {}
    # dictStndrd = {}
    # dictRobust = {}
    # dictQuantN = {}
    # dictNoScle = {}
    allDict = { 
                # "b0":{ "standard":{}, "minmax":{}, "maxabs":{}, "noscale":{} }, 
                # "b1":{ "standard":{}, "minmax":{}, "maxabs":{}, "noscale":{} }, 
                # "b2":{ "standard":{}, "minmax":{}, "maxabs":{}, "robust":{}, "noscale":{} }, 
                # "q0":    { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q1":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q2":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q3":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "b0_ns":{ "standard":{}, "minmax":{}, "maxabs":{}, "noscale":{} }#, 
                # "q0_10k":{ "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q1_10k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q2_10k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q3_10k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q0_100k":{ "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q1_100k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q2_100k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q3_100k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} },        
                # "q0_100k":{ "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q1_100k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q2_100k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q3_350k":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} },
                # "q0_1M":{ "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q1_1M":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q2_1M":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q3_1M":{ "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }                           
                # "qpxy":      { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qm":        { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qe":        { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q0_10k":    { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} },
                # "qpxy_10k":  { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qm_10k":    { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qe_10k":    { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q0_100k":   { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} },
                # "qpxy_100k": { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qm_100k":   { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qe_100k":   { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q0_350k":   { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} },
                # "qpxy_350k": { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qm_350k":   { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qe_350k":   { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "q0_1M":     { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} },
                # "qpxy_1M":   { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qm_1M":     { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                # "qe_1M":     { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} },
                # "u0":{ "standard":{}, "minmax":{}, "maxabs":{}, "quantu":{}, "noscale":{} }, 
                # "u1":{ "minmax":{}, "maxabs":{}, "quantu":{}, "noscale":{} }, 
                # "u2":{ "minmax":{}, "maxabs":{}, "quantu":{}, "noscale":{} }, 
                # "u3":{ "minmax":{}, "maxabs":{}, "quantu":{}, "noscale":{} }
                "Basic": { "standard":{}, "minmax":{}, "maxabs":{},              "noscale":{} }, 
                "Qmpxy": { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                "Qall":  {                "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }

              }
    # quantDict = {"n_quantiles":{"q0_1M":1000000, "q1_1M":1000000, "q2_1M":1000000, "q3_1M":1000000},
    #              "n_jobs":{"q0_-1} }
    bigKeys = allDict.keys()
    bigKeys.sort()
    print(bigKeys)
    i = 0
    
    #####check if no Scale SV (b0_ns) is better or not

    for line in maskFile:
        index, var = line.split(':')
        var = var.strip()
        varDict[index] = var
        revDict[var] = index
        allVars.append(var)

        if "jetAK8_pt" in var:
            for bigDict in allDict.values(): bigDict["minmax"][i] = (var) 
        elif "jet_energy" in var:
            for bigKey, bigDict in allDict.items():
                if ("Qall" in bigKey): bigDict["quantn"][i] = (var)
                else:                  bigDict["minmax"][i] = (var)
                # if ("q2" in bigKey) or ("q3" in bigKey): bigDict["quantn"][i] = (var)
                # if   ("q2" in bigKey) or ("q3" in bigKey) or ("qe" in bigKey):  bigDict["quantn"][i] = (var)
                # elif ("u2" in bigKey) or ("u3" in bigKey):  bigDict["quantu"][i] = (var)
                # else:                                       bigDict["minmax"][i] = (var)
        elif "jet_pz" in var:
            for bigKey, bigDict in allDict.items():
                if ("Qall" in bigKey): bigDict["quantn"][i] = (var)
                else:                  bigDict["standard"][i] = (var)                
                # if   ("b" in bigKey) or ("qpxy" in bigKey) or ("qm" in bigKey) or ("qe" in bigKey): bigDict["standard"][i] = (var)
                # elif ("u" in bigKey): bigDict["quantu"][i]   = (var)
                # else:                 bigDict["quantn"][i]   = (var)            
        elif "jet_p" in var:
            for bigKey, bigDict in allDict.items():
                if ("Q" in bigKey): bigDict["quantn"][i] = (var)
                else:               bigDict["standard"][i] = (var)  
                # if    ("b" in bigKey) or ("q0" in bigKey) or ("u0" in bigKey) or ("qm" in bigKey) or ("qe" in bigKey): bigDict["standard"][i] = (var)
                # elif  ("u" in bigKey):                                             bigDict["quantu"][i]   = (var)
                # else:                                                              bigDict["quantn"][i]   = (var)
        # elif "nJets_" in var:
        #     for bigDict in allDict.values(): bigDict["maxabs"][i] = (var) 
        # elif "nSecondary" in var:
            # for bigDict in allDict.values(): bigDict["maxabs"][i] = (var) 
        elif "mass" in var.lower():
            for bigKey, bigDict in allDict.items():
                if ("Q" in bigKey): bigDict["quantn"][i] = (var)
                else:               bigDict["maxabs"][i] = (var)  
                # if   ("b2" in bigKey): bigDict["robust"][i] = (var)
                # elif ("q" in bigKey):  bigDict["quantn"][i] = (var)
                # # elif ("q3" in bigKey) or ("qm" in bigKey): bigDict["quantn"][i] = (var)
                # else:                  bigDict["maxabs"][i] = (var)    
        elif "jetAK8_eta" in var:        
            for bigKey, bigDict in allDict.items(): bigDict["maxabs"][i] = (var)
            # for bigKey, bigDict in allDict.items():
            #     if   ("b0" == bigKey): bigDict["noscale"][i] = (var)
            #     else:                  bigDict["maxabs"][i] = (var)
        else: 
            for bigDict in allDict.values(): bigDict["noscale"][i] = (var)
        i += 1
    maskFile.close()
    del i        
    mask = [True if str(i) in varDict else False for i in range(596)]
    #######run as is, making a new version that doesnt scale nSec. compare to og b0.
    # new batch: no scaling nsec. max abs scale eta

    # after this, select whether or not we scale eta and nsec. do the best ones.
    # then, run the quantile with way more quantiles. plot. do a few different ones (10,000, 100,000)
    # sub in quantile uniform for quantile normal.
    # create the  uniform, and updated normal models at the same time.       
        # quantNBEST
        # if ("jet_energy" in var) or ("jet_p" in var) or ("nJets" in var) or ("nSecondary" in var) or ("mass" in var) or ("jetAK8_eta" in var) : dictQuantN[index] = (var)
        # elif ("jetAK8_pt" in var): dictMinMax[index] = (var)

        # robustBEST
        # if ("jet_energy" in var) or ("jet_p" in var) or ("nJets" in var) or ("nSecondary" in var)or ("mass" in var) : dictRobust[index] = (var)
        # elif ("jetAK8_eta" in var) or ("jetAK8_pt" in var): dictMinMax[index] = (var)

        # basicBEST
        # if ("jet_energy" in var) or ("jetAK8_pt" in var): dictMinMax[index] = (var)
        # elif ("jet_pz" in var): dictStndrd[index] = (var)
        # elif ("jet_pz" in var) or ("jetAK8_eta" in var): dictStndrd[index] = (var)
        # elif ("jet_p" in var) or ("mass" in var) or ("nJets" in var) or ("nSecondary" in var): dictMaxAbs[index] = (var)
        # elif ("jet_p" in var) or ("mass" in var) or ("nJets" in var) or ("nSecondary" in var) or ("jetAK8_eta" in var): dictMaxAbs[index] = (var)

        # else: dictNoScle[index] = (var)
    

    # for bigKey, bigDict in allDict.items():
    # for bigKey in bigKeys:
    #     print(bigKey)
    #     bigDict = allDict[bigKey]
    #     for myScale, myDict in bigDict.items():
    #         print(myScale, myDict)
    #         print("")
    #     print('\n')
    # print('\n\n')

    transDict = {}

    # for bigKey, bigDict in allDict.items(): # bigKey,bigDict = "b0",{"minmax"....}
    for bigKey in bigKeys:
        # if not ( (bigKey == "q2") or (bigKey == "q3") ): continue  
        # if not ( bigKey == 'b0' ): continue  
        bigDict = allDict[bigKey]
        # print(bigKey)
        transDict[bigKey] = []
        i = 0
        while i < len(allVars):
            var = allVars[i]
            # trueIndex = revDict[var]
            # print(i,trueIndex,var)

            for myScale, myDict in bigDict.items(): #myScale,myTrans = "minmax", {index:(var), ....}
                # print(myScale, myDict)
                # print(myDict.values())
                # ivd = {v: k for k, v in myDict.items()}
                if not (var in myDict.values()): continue
                varKeys = list( map(int, myDict.keys()) ) # this is a list of the allVars index for this mask for this ct for this scaler, sorted.
                varKeys.sort()
                repeats = checkRepeat(i, varKeys)
                transName = bigKey + '_' + myScale + '_' + str(i)
                # print(transName, repeats)
                nquant = 350000
                # nquant = 1000
                # if   "10k"  in bigKey: nquant = 10000
                # elif "100k" in bigKey: nquant = 100000
                # elif "350k" in bigKey: nquant = 350000
                # elif "1M"   in bigKey: nquant = 1000000
                transformers = transDict[bigKey]
                if myScale == "standard":  transformers.append( (transName, preprocessing.StandardScaler(), repeats) )  
                elif myScale == "minmax":  transformers.append( (transName, preprocessing.MinMaxScaler(), repeats) )
                elif myScale == "maxabs":  transformers.append( (transName, preprocessing.MaxAbsScaler(), repeats) )
                elif myScale == "robust":  transformers.append( (transName, preprocessing.RobustScaler(), repeats) ) 
                elif myScale == "quantn":  transformers.append( (transName, preprocessing.QuantileTransformer(n_quantiles = nquant, subsample = 20000000, output_distribution = 'normal'), repeats) )  
                # elif myScale == "quantn":  transformers.append( (transName, preprocessing.QuantileTransformer(n_quantiles = nquant, subsample = 9000000, output_distribution = 'normal'), repeats) )  
                # elif myScale == "quantn":  transformers.append( (transName, preprocessing.QuantileTransformer(n_quantiles = 1000000, output_distribution = 'normal'), repeats) )  
                elif myScale == "quantu":  transformers.append( (transName, preprocessing.QuantileTransformer(output_distribution = 'uniform'), repeats) )  
                elif myScale == "noscale": transformers.append( (transName, "passthrough", repeats) )
                
                else: 
                    print("bad thing happened")
                    quit()
            i += len(repeats)

    # for bigKey in bigKeys:
    #     print(bigKey + '\n\n\n\n\n')
    #     for trans in transDict[bigKey]:
    #         begin = trans[0].rfind('_') + 1
    #         print(allVars[int(trans[0][begin:])], len(trans[2]), trans)
    #         print("")
    # quantiles = 350000
    mySet = "train"
    # mySet = "test"
    print("Loading pre scale h5py files")
    # preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"])[()] for mySample in sampleTypes]
    preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"])[:,mask] for mySample in sampleTypes]
    print("Pre scale events shape:", [arr.shape for arr in preScaleEvents])
    # plotBESVars(preScaleEvents, sampleTypes, "preScale", plotDir, mySet, allVars, plotlist)
    # del preScaleEvents
    print("Concatenating...")
    preScaleAll = np.concatenate(preScaleEvents)
    del preScaleEvents
    print("Pre scale events shape:", preScaleAll.shape)

    print("Starting loop...")    
    startTime = time.time()
    for bigKey in bigKeys:
        # if not ( (bigKey == "q2") or (bigKey == "q3") ): continue  
        # if not ( "M" in bigKey ): continue  

        loopTime = time.time()
        print(bigKey)

        # if ("q2" in bigKey) or ("q3" in bigKey): njobs = None
        # if ("Qall" in bigKey): njobs = None
        # else:                  njobs = -1
        njobs = None
        ct = ColumnTransformer(
            transDict[bigKey], #transformer list
            remainder = "drop",
            n_jobs = njobs
        )
        # print(ct)
        # print("")
        # print(ct.get_params())
        print("")
        for key, val in allDict[bigKey].items(): 
            print(key, val)
            print("")

        # print(allDict[bigKey])
        # print("")

        midTime = time.time()
        print("fitting...")
        ct.fit(preScaleAll)
        # ct = load("/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/basicBEST.joblib")
        midTime2 = time.time()
        print("done fitting; minutes: ", (midTime2 - midTime)/60. ) 

        # print("transforming...")
        # postScaleAll = ct.transform(preScaleAll)
        # midTime3 = time.time()
        # print("done transforming; minutes: ", (midTime3 - midTime2)/60. )

        # if not preScaleAll.shape == postScaleAll.shape: 
        #     print("errrror var lost", preScaleAll.shape, postScaleAll.shape)
        #     quit()

        saveScale = 'ScalerParameters/newBEST_' + bigKey + '.joblib'
        print("dump: " + saveScale)
        dump(ct, saveScale)
        del ct
        # print("done with " + bigKey + "loop; minutes: ", (time.time() - loopTime)/60.)
    del preScaleAll
    print("done in total; minutes: ", (time.time() - startTime)/60.)
    quit()

    #       standard: pz*, pxpy*
    #       minmax: ak8pt, energy*
    #       maxAbs: nJets_frame (don't scale njets), nSec**, eta (or no scale), mass*
    #       robust: nSec*, mass*
    #       quantN: pz**, pxpy*, energy*, mass*      
    #       make scales for oldBEST, then loop over them and compare.
    #       NOTE: OLD BEST DOES NOT USE MOST OF THE SCALED VARS
    #       NOTE: NEW BEST DOES NOT USE nJet, SO TEST IT LATER
    # b0:
    #       standard: pz, pxpy
    #       minmax: ak8pt, energy
    #       maxAbs: nJets_frame (don't scale njets), mass
    #       do not scale eta
    # b1:
    #       standard: pz, pxpy
    #       minmax: ak8pt, energy
    #       maxAbs: nJets_frame (don't scale njets), eta, mass
    # b2:
    #       standard: pz, pxpy
    #       minmax: ak8pt, energy
    #       maxAbs: nJets_frame (don't scale njets), eta
    #       robust: mass      
    # q0:
    #       standard: pxpy*
    #       minmax: ak8pt, energy*
    #       maxAbs: nJets_frame (don't scale njets), eta, mass*
    #       quantN: pz
    # q1:
    #       minmax: ak8pt, energy*
    #       maxAbs: nJets_frame (don't scale njets), eta, mass*
    #       quantN: pz, pxpy
    # q2:
    #       minmax: ak8pt
    #       maxAbs: nJets_frame (don't scale njets), eta, mass*
    #       quantN: pz, pxpy, energy
    # q3:
    #       minmax: ak8pt
    #       maxAbs: nJets_frame (don't scale njets), eta
    #       quantN: pz, pxpy, energy, mass
    #  
    #       jet _energy scale to 0,1. minmax, or robust/quant
    #       jet_pxpy scale -1,1, maxabs, standard(center on/off)?  robust?     
    #       jet pz needs testing. robust/quantile, standardd, minmax
    #       eta use gaussian scaler. minmax or standard, look at output from both (plot)
    #       njets/nSecVert, robust?quantile? maxabs?
    #       check with and without phi later
    #       ak8pt scale to 0,1, minmax
    """
    """
    # maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_noDeepAK8noNJets_trimIso.txt"
    maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/allBESTVars.txt"
    maskFile = open(maskPath, "r")
    allVars = []
    varDict = {}
    dictMinMax = {}
    dictMaxAbs = {}
    dictStndrd = {}
    dictRobust = {}
    dictQuantN = {}
    dictNoScle = {}
    plotlist = ["204", "208", "216", "219", "232", "235", "380", "425", "470" ]

    for line in maskFile:
        index, var = line.split(':')
        var = var.strip()
        varDict[index] = var
        allVars.append(var)


        if index in plotlist: dictQuantN[index] = (var)

        # quantNBEST
        # if ("jet_energy" in var) or ("jet_p" in var) or ("nJets" in var) or ("nSecondary" in var) or ("mass" in var) or ("jetAK8_eta" in var) : dictQuantN[index] = (var)
        # elif ("jetAK8_pt" in var): dictMinMax[index] = (var)

        # robustBEST
        # if ("jet_energy" in var) or ("jet_p" in var) or ("nJets" in var) or ("nSecondary" in var)or ("mass" in var) : dictRobust[index] = (var)
        # elif ("jetAK8_eta" in var) or ("jetAK8_pt" in var): dictMinMax[index] = (var)

        # basicBEST
        # if ("jet_energy" in var) or ("jetAK8_pt" in var): dictMinMax[index] = (var)
        # elif ("jet_pz" in var): dictStndrd[index] = (var)
        # elif ("jet_pz" in var) or ("jetAK8_eta" in var): dictStndrd[index] = (var)
        # elif ("jet_p" in var) or ("mass" in var) or ("nJets" in var) or ("nSecondary" in var): dictMaxAbs[index] = (var)
        # elif ("jet_p" in var) or ("mass" in var) or ("nJets" in var) or ("nSecondary" in var) or ("jetAK8_eta" in var): dictMaxAbs[index] = (var)

        else: dictNoScle[index] = (var)
    maskFile.close()

    # mask = [True if str(i) in varDict else False for i in range(596)]
    #######run as is, making a new version that doesnt scale nSec. compare to og b0.
    # after this, select whether or not we scale eta and nsec. do the best ones.
    # then, run the quantile with way more quantiles. plot. do a few different ones (10,000, 100,000)
    # sub in quantile uniform for quantile normal.
    # create the uniform, and updated normal models at the same time.       
    
    # for i in range(596):
    #     var = varDict[str(i)]
    #     doubFlag = False
    #     if var in dictMinMax.values(): 
    #         print(i,var,"MinMax")
    #         if doubFlag: print(i,var,"\nNOOOOOOOOOOOO\n\n\n\n\n\n\nAHHHHHHH\n",i, var)
    #         doubFlag = True
    #     if var in dictMaxAbs.values(): 
    #         print(i,var,"MaxAbs")
    #         if doubFlag: print(i,var,"\nNOOOOOOOOOOOO\n\n\n\n\n\n\nAHHHHHHH\n",i, var)
    #         doubFlag = True
    #     if var in dictStndrd.values(): 
    #         print(i,var,"Standard")
    #         if doubFlag: print(i,var,"\nNOOOOOOOOOOOO\n\n\n\n\n\n\nAHHHHHHH\n",i, var)
    #         doubFlag = True
    #     if var in dictNoScle.values():
    #         if doubFlag: print(i,var,"\nNOOOOOOOOOOOO\n\n\n\n\n\n\nAHHHHHHH\n",i, var)
    #         doubFlag = True

    # print("Not used vars:")
    # for i in range(596):
    #     var = varDict[str(i)]
    #     if str(i) in dictNoScle: print(i,var)
    # print("MaxAbs: ",len(dictMaxAbs), dictMaxAbs)
    # print("MinMax: ",len(dictMinMax), dictMinMax)
    # print("Standard: ",len(dictStndrd), dictStndrd)
    # print("NoScale: ",len(dictNoScle), dictNoScle)
    # print(len(dictMaxAbs) + len(dictMinMax) + len(dictStndrd) + len(dictNoScle))
    # keysMinMax = list(map(int, dictMinMax.keys())) ; keysMinMax.sort()
    # keysMaxAbs = list(map(int, dictMaxAbs.keys())) ; keysMaxAbs.sort()
    # keysStndrd = list(map(int, dictStndrd.keys())) ; keysStndrd.sort()
    # keysRobust = list(map(int, dictRobust.keys())) ; keysRobust.sort()
    keysQuantN = list(map(int, dictQuantN.keys())) ; keysQuantN.sort()
    keysNoScle = list(map(int, dictNoScle.keys())) ; keysNoScle.sort()
    # for key in keysNoScle:
    #     print(key,dictNoScle[str(key)])

    # quit()
    # plotlist = keysMinMax[0:5] + keysMaxAbs[0:5] + keysStndrd[0:5] + keysNoScle[0:5]
    # plotlist = keysMinMax[:] + keysMaxAbs[:] + keysStndrd[:] + keysNoScle[:]
    # plotlist = [140, 143, 204, 208, 216, 219, 232, 235, 380, 425, 470, 515, 545, 548, 549, 559, 565 ]
    # plotlist = [140, 143, 204, 208, 216, 219, 232, 235, 380, 425, 470, 515, 545, 549, 559, 565 ]
    plotlist = [204, 208, 216, 219, 232, 235, 380, 425, 470 ]
    # plotlist = [545]
    # print(plotlist)
    # print("minmax: ", keysMinMax)
    # print("maxabs: ", keysMaxAbs)
    # print("standard: ", keysStndrd)
    # print("no scale: ", keysNoScle)
    # quit()
#could make a list of strings, for each var, of which scaler

    plotDir = "plots/BESvars_scaleCompare/"
    # arr = np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs.h5"["BES_vars"][:,mask],"r")
    # arrays  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs.h5","r")["BES_vars"])[:,mask] for mySample in sampleTypes]
    # arrays  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"][()]) for mySample in sampleTypes]
    # arrays  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"]) for mySample in sampleTypes]
    
    transformers = []
    # for i in range(596):
    i = 0
    #make scaler maker? like in load scaler, update it to handle ct

    while i < 596: #len(mask)

        ind = str(i)
        var = varDict[ind]
        # print('\n')
        # print(ind,var)
        # if i in keysMinMax:   transformers.append("MinMax")
        # elif i in keysMaxAbs: transformers.append("MaxAbs")
        # elif i in keysStndrd: transformers.append("Stndrd")
        # elif i in keysNoScle: transformers.append("NoScle")
        # if i in keysMinMax:   
        #     # print( 'MinMax: ')
        #     repeats = checkRepeat(i,keysMinMax)
        #     transformers.append( ("MinMax_"+str(i), preprocessing.MinMaxScaler(), repeats) )
        #     i += len(repeats)
        # elif i in keysMaxAbs: 
        #     # print( 'MaxAbs: ')
        #     repeats = checkRepeat(i,keysMaxAbs)
        #     transformers.append( ("MaxAbs_"+str(i), preprocessing.MaxAbsScaler(), repeats) )
        #     i += len(repeats)
        # elif i in keysStndrd: 
        #     # print( 'Standard: ')
        #     repeats = checkRepeat(i,keysStndrd)
        #     transformers.append( ("Stndrd_"+str(i), preprocessing.StandardScaler(), repeats) )          
        #     i += len(repeats)
        # elif i in keysRobust: 
        #     # print( 'Standard: ')
        #     repeats = checkRepeat(i,keysRobust)
        #     transformers.append( ("Robust_"+str(i), preprocessing.RobustScaler(), repeats) )          
        #     i += len(repeats) 
        if i in keysQuantN: 
            # print( 'Standard: ')
            repeats = checkRepeat(i,keysQuantN)
            transformers.append( ("QuantN_"+str(i), preprocessing.QuantileTransformer(n_quantiles = 10000, output_distribution = 'normal'), repeats) )          
            i += len(repeats)         
        elif i in keysNoScle: 
            # print( 'No Scale: ')
            repeats = checkRepeat(i,keysNoScle)
            transformers.append( ("NoScle_"+str(i), "passthrough", repeats) )
            i += len(repeats)
        else: 
            print("bad thing happened")
            quit()



    #       jet _energy scale to 0,1. minmax, or robust/quant
    #       jet_pxpy scale -1,1, maxabs, standard(center on/off)?  robust?     
    #       jet pz needs testing. robust/quantile, standardd, minmax
    #       eta use gaussian scaler. minmax or standard, look at output from both (plot)
    #       njets/nSecVert, robust?quantile? maxabs?
    #       check with and without phi later
    #       ak8pt scale to 0,1, minmax
    #######
    # mySet = "test"
    mySet = "train"
    print("Loading pre scale h5py files")
    preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"])[()] for mySample in sampleTypes]
    # preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"])[:,mask] for mySample in sampleTypes]
    print("Pre scale events shape:", [arr.shape for arr in preScaleEvents])
    # plotBESVars(preScaleEvents, sampleTypes, "preScale", plotDir, mySet, allVars, plotlist)
    # del preScaleEvents
    # scales = [ "StandardScaler", "MinMax01Scaler", "MinMax11Scaler", "MaxAbsScaler",
    #                  "RobustScaler", "QuantileTransformerNormal",  "QuantileTransformerUniform" ]

    
    startTime = time.time()
    ct = ColumnTransformer(
        transformers, #transformer list
        remainder = "drop",
        n_jobs = -1 #test quantileN on test set with this on and off
    )

    preScaleAll =  np.concatenate(preScaleEvents)
    midTime = time.time()
    print("Pre scale events shape:", preScaleAll.shape)
    print("Concatenated ", midTime - startTime)
    ct.fit(preScaleAll)
    del preScaleAll
    # ct = load("/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/basicBEST.joblib")
    midTime2 = time.time()
    print("done fitting ", midTime2 - midTime)

#need to fix plotter for indices, idr what i did before 
    # postScaleEvents = ct.transform(preScaleEvents)
    postScaleEvents = [ct.transform(arr) for arr in preScaleEvents]
    print("done transforming ", time.time() - midTime2 )

    if not preScaleEvents[0].shape == postScaleEvents[0].shape: 
        print("errrror var lost", preScaleEvents.shape, postScaleEvents.shape)
        quit()
    del preScaleEvents

    print("plotting postscale")    
    plotBESVars(postScaleEvents, sampleTypes, "postScaleQuantN_test_10k", plotDir, mySet, allVars, plotlist)
    del postScaleEvents

    # print("dump")
    # dump(ct, 'ScalerParameters/basicEtaBEST.joblib')
    del ct
    quit()
    """    

    # ctnew = load("/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/bestTransformer.joblib")
    # newpostScale = [ctnew.transform(arr) for arr in preScaleEvents]
    # del preScaleEvents
    # plotBESVars(newpostScale, sampleTypes, "postScaleLoad", plotDir, mySet, allVars, plotlist)

    # print("compare matrices")
    # for i, samp in enumerate(sampleTypes):
    #     if np.array_equal(postScaleEvents[i], newpostScale[i]): print(samp + " is equal!")

    # del newpostScale


    # print("i,prescale,postscale")
    # for i in range(596):
        # print(i, preScaleEvents[0][:5,i], postScaleEvents[0][:5,i])

    # print("These should not be scaled:")
    # for i in keysNoScle: 
    # #     print(i)
    #     if not (preScaleEvents[:,i] == postScaleEvents[:,i]).all(): quit()

    #       jet _energy scale to 0,1. minmax, or robust/quant
    #       jet_pxpy scale -1,1, maxabs, standard(center on/off)?  robust?     
    #       jet pz needs testing. robust/quantile, standardd, minmax
    #       eta use gaussian scaler. minmax or standard, look at output from both (plot)
    #       check with and without phi later
    #       ak8pt scale to 0,1, minmax
    #       njets/nSecVert, robust?quantile? maxabs?

    # transformers = [("MinMax", preprocessing.MinMaxScaler(), ["jet energy all frames 0,1,2,3, ak8pt "]),   #list of tuples (name, transformer, column(s))
    #                 ("MaxAbs",preprocessing.MaxAbsScaler(),["jet px py all frames 0123, mass, njets all frames,nsecvert"])
    #                 ("Standard",preprocessing.StandardScaler(),["jet pz all frames 0123, eta"]),
    #                 ("NoScale", "passthrough",["all else"])
    # ]

    # ==================================================================================
    # Standardize BES Vars ////////////////////////////////////////////////////////////
    # ==================================================================================
    
    # scales = [ "StandardScaler", "MinMax01Scaler", "MinMax11Scaler", "MaxAbsScaler", "RobustScaler", "QuantileTransformerNormal",  "QuantileTransformerUniform" ]
    # scales = [ "StandardScaler" ]
    # for scale in scales:

    #     scalePath = "ScalerParameters/ScalerParameters_" + scale + ".joblib"
    #     # Load scaler model  
    #     scaler = tools.loadScalerModel(scalePath, mask)

    #     print("Scaling...")
    #     postScaleTrainEvents = [scaler.transform(arr) for arr in preScaleEvents]
    #     del preScaleEvents
    #     print("postScaleTrain events shape:",[arr.shape for arr in postScaleTrainEvents])
    #     plotBESVars(postScaleTrainEvents, sampleTypes, "postScaleTrain", plotDir, mySet, allVars, mask)

    #     print("Scaling with Train...")
    #     postScaleTrainEvents = [scaler.transform(arr) for arr in preScaleEvents]
    #     del preScaleEvents
    #     print("postScaleTrain events shape:",[arr.shape for arr in postScaleTrainEvents])
    #     plotBESVars(postScaleTrainEvents, sampleTypes, "postScaleTrain", plotDir, mySet, allVars, mask)

    #     print("Unscaling with Train...")
    #     unScaleTrain = [scaler.inverse_transform(arr) for arr in postScaleTrainEvents]
    #     print("Unscaled runtime events shape:",[arr.shape for arr in unScaleTrain])
    #     plotBESVars(unScaleTrain, sampleTypes, "unScaledTrain", plotDir, mySet, allVars, mask)
    #     del unScaleTrain

    #     allBESinputs = np.concatenate(preScaleEvents)
    #     print("Shape allBESinputs", allBESinputs.shape)
    #     scaler = preprocessing.StandardScaler().fit(allBESinputs)
    #     del allBESinputs

    #     print("Saving Model....")
    #     scalePath = 'ScalerParameters_' + mySet + '.txt'
    #     with open(scalePath, 'w') as outputFile:
    #         for mean,var in zip(scaler.mean_, scaler.var_):
    #             outputFile.write('{},{}\n'.format(mean, var))
        
    #     dump(scaler, 'ScalerParameters_' + mySet + '.joblib')
        
    #     print("Scaling...")
    #     postScaleEvents = [scaler.transform(arr) for arr in preScaleEvents]
    #     del preScaleEvents
    #     print("postScale events shape:",[arr.shape for arr in postScaleEvents])
    #     plotBESVars(postScaleEvents, sampleTypes, "postScale" + mySet, plotDir, mySet, allVars, mask)

    #     print("Unscaling with runtime model...")
    #     unScaleRun = [scaler.inverse_transform(arr) for arr in postScaleEvents]
    #     print("Unscaled runtime events shape:",[arr.shape for arr in unScaleRun])
    #     plotBESVars(unScaleRun, sampleTypes, "unScaled" + mySet, plotDir, mySet, allVars, mask)
    #     del unScaleRun
        
    #     print("Unscaling with loaded model from joblib...")
    #     del scaler
    #     scaler = load('ScalerParameters_' + mySet + '.joblib')
    #     unScaleLoadJoblib = [scaler.inverse_transform(arr) for arr in postScaleEvents]
    #     print("Unscaled loaded means events shape:",[arr.shape for arr in unScaleLoadJoblib])
    #     plotBESVars(unScaleLoadJoblib, sampleTypes, "unScaledLoadedJoblib", plotDir, mySet, allVars, mask)
    #     del unScaleLoadJoblib

    #     print("Unscaling with loaded model from means...")
    #     del scaler
    #     scaleFile = open(scalePath, "r")
    #     means  = []
    #     scales = []
    #     for line in scaleFile:
    #         mean, vari = line.split(',')
    #         vari = vari.strip()
    #         means.append(float(mean))
    #         scales.append(np.sqrt(float(vari)))
    #     scaleFile.close()

    #     scaler = preprocessing.StandardScaler()
    #     scaler.mean_ = np.array(means)
    #     scaler.scale_ = np.array(scales)
    #     unScaleLoadMeans = [scaler.inverse_transform(arr) for arr in postScaleEvents]
    #     del postScaleEvents
    #     print("Unscaled loaded means events shape:",[arr.shape for arr in unScaleLoadMeans])
    #     plotBESVars(unScaleLoadMeans, sampleTypes, "unScaledLoadedMeans", plotDir, mySet, allVars, mask)
    #     del unScaleLoadMeans

    # print("nJets events shape:", [arr.shape for arr in arrays])
    # plotBESVars(arrays, sampleTypes, "", plotDir, "", allVars, mask)

    # outFile = open("checkRanges_4VPF.txt", "w")

    # for index, var in enumerate(allVars):
    #     if not mask[index]: continue
    #     var = varDict[str(index)]
    #     print(var)
    #     outFile.write( var + ":\n")

    # for iarr, array in enumerate(arrays):
    #     samp = sampleTypes[iarr]
    #     print(samp)
    #     outFile.write(samp + ":\n")
    #     print("bes shape: ",h5py.File(h5Dir+samp+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"].shape)
    #     # arr = array[:,index]
    #     for ev in range(10):
    #         vec = array[ev,:]
    #     # arr = arr[arr<4]
    #     # print(array.shape)
    #     # print(arr.shape)
    #     # if len(arr) < 1: continue
    #         # pfl = h5py.File(h5Dir+samp+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["PF_cands_LabFrame"][ev,:,3] #3 = energy pf cand 
    #         pfl = h5py.File(h5Dir+samp+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["PF_cands_LabFrame"] #3 = energy pf cand 
    #         print("lab pf shape: ", pfl.shape)
    #         pfl = pfl[ev,:,3]
    #         # print(pfl)
    #         # print(pfl.shape)
    #         # print(pfh.shape)
    #         # print(pfh)
    #         outFile.write("\nevent: " + str(ev) + ", LabFrame PF Cands: " + str(len(pfl[pfl > 0])) + "\n")
    #         for i, frame in enumerate(frameTypes):
    #             # pfs = h5py.File(h5Dir+samp+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["PF_cands_"+frame+"Frame"][ev,:,3]
    #             pfs = h5py.File(h5Dir+samp+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["PF_cands_"+frame+"Frame"]
    #             print(frame + " pf shape: ", pfs.shape)
    #             pfs = pfs[ev,:,3]
    #             outFile.write("\tframe: " + frame + ", PF Cands: " + str(len(pfs[pfs > 0])) + "\n\t\tsubjet_index: [e  px  py  pz ]\n")

    #             for k in range(4):
    #                 ind = i*12 + k 
    #                 tempMask = [False for x in range(80)]
    #                 for j in range(4): tempMask[i*16 + j*4 + k] = True 
    #                 # tempmask = [True if  else False for x in range(4)]
    #                 outFile.write("\t\t" + str(k) + ": " + str(vec[tempMask]) + "\n")
    # outFile.write("\n")
    # outFile.close()
    # quit()

    # outFile = open("checkRanges_PFLABWW366818.txt", "a")
    # outFile.write("event: 366818;   PF_cands_LabFrame\n")
    # f  = h5py.File(h5Dir+"WWSample_2017_BESTinputs_test_flattened.h5","r")
    # keys = list(f.keys())
    # print(keys)
    # bv = f["BES_vars"]
    # # print(bv[366818].shape)
    # # print(bv[366818,410])
    # # pf = f["PF_cands_HiggsFrame"]
    # pf = f["PF_cands_LabFrame"]
    # print(pf.shape) # events x pfcands x var
    # for i in range(11):
    #     arr = pf[336818,:,i]
    #     # print(arr.shape)
    #     var = varDict[str(i)]
    #     # print(var)
    #     outFile.write(str(i) + "," + var + ":\n")
    #     outFile.write(str(arr))
    #     outFile.write('\n')

    # outFile.close()
    # quit()

    # for i in range(events.shape[0]):
    # badEvent = np.argmin(events[:,410])
    # print(events.shape)
    # print(varDict["410"])
    # print(badEvent)
    # outFile.write("event: " + str(badEvent) + "\n")
    # arr = events[badEvent,:]
    # print(arr.shape)
    # for j in range(len(allVars)):
    #     var = varDict[str(j)]
    #     outFile.write(str(j) + "," + var + ": " + str(arr[j]) + "\n")
    # quit()

        # event = events[i,410]
        # if event == -999.99: 
        #     outFile.write("event: " + str(i) + "\n")
        #     for j in range(len(allVars)):
        #         outFile.write(str(index) + "," + var + ": " + str(events[i,j]) + "\n")
        #     quit()


    # for index, var in varDict.items():
    # for index in range(len(allVars)):
    #     var = varDict[str(index)]
    #     # array = preScaleDict["trainEvents"][:,index]
    #     array = preScaleDict["testEvents"][:,index]
    #     min = np.min(array)
    #     max = np.max(array)
    #     fmax = np.fabs(max)
    #     fmin = np.fabs(min)
    #     if fmax > 10.0: continue
    #     # if fmin > 10.0: continue
    #     if (np.fabs(fmax - fmin) > 1.0 ): continue
    #     # if (np.fabs(fmax - fmin) > 1.0 ): outFile.write(str(index) + "," + var + ": min = " + str(min) + " max = " + str(max) + "\n")
    #     outFile.write(str(index) + "," + var + ": min = " + str(min) + " max = " + str(max) + "\n")
    # outFile.close()

    # print("Pre scale events shape:", [arr.shape for arr in preScaleDict])
    # plotBESVars(preScaleDict, sampleTypes, "preScale", plotDir, mySet, allVars, mask)



    # for mySet in setTypes:
    #     print("Beginning " + mySet)
    #     plotDir = "plots/BESvars_checkRangeTrain/"





    # Check how long the script took to run
    # timelog = open("Logs/scaleTest_TimeLog.txt", "a") 
    timeTaken = divmod(time.time() - startTime, 60.)
    timeMessage = "Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
    print(timeMessage)
    # timelog.write(timeMessage)
    # timelog.close

