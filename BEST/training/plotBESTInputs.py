#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trainBEST.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST: The Boosted Event Shape Tagger ////////////////////////
#==================================================================================

#This script needs to be cleaned up still
# has a lot of scratch works. needs to be updated so it can be part of the normal pipeline

# user modules
import tools.functions as tools
startTime = tools.logTime() # Tracks how long script takes

# modules
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import os


# get stuff from modules
# from sklearn import preprocessing


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
"""
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask.txt"
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_noDeepAK8noNJets_trimIso.txt"
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/h5samples/SVvarList.txt"
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/h5samples/pfDEPvarList.txt"
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/h5samples/pfINVvarList.txt"
# maskFile = open(maskPath, "r")
# varDict = {}
# allVars = []
# for line in maskFile:
#     index, var = line.split(':')
#     var = var.strip()
#     varDict[index] = var
#     allVars.append(var)
# maskFile.close()
# print(maskPath + " chosen; mask size " + str(len(allVars)))
# # myMask = [True if str(i) in varDict else False for i in range(596)]
"""

maskPath = "/uscms_data/d3/cannaert/BEST/CMSSW_10_6_27/src/BEST/formatConverter/h5samples/BESvarList.txt"
mask, varDict = tools.loadMask(maskPath)

# sampleFileTypes = ["WW","ZZ","HH","TT","BB","QCD"]
sampleFileTypes = ["Signal","QCD","TTBar"]
samples     = ["Signal","QCD","TTBar"]
# sampleFileTypes = ["HH"]
# samples     = ["H"]
# h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
h5Dir = "/uscms_data/d3/cannaert/BEST/CMSSW_10_6_27/src/BEST/formatConverter/h5samplesSplit/"

# years = ["2016_APV","2016","2017","2018"]
# years = ["2016_APV","2016"]
years = ["2018"]


# setTypes = ["validation","test","train"]
setTypes = ["train"]
# setTypes = ["test"]
suffix = "flattened"
#suffix = "flattened"
# suffix = "flatTop"

# This code plots the pdgids, and loads a mask for padded entries
"""
h5MaskKey = "PF_cands_LabFrame"
h5Key = "PF_cands_AllFrame"
print("Loading h5 mask var... ")
maskVar = 2
maskData = {}
for setType in setTypes:
    print("Loading " + setType)
    maskData[setType] = {}
    for sample in sampleFileTypes:
        # maskData[setType][sample] = np.array(h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + setType + "_flattened.h5","r")[h5MaskKey][:,:,maskVar])
        maskData[setType][sample] = np.array(h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + setType + ".h5","r")[h5MaskKey][:,:,maskVar])

    
for setType in setTypes: print("Initial " + setType + " mask shape: ", [maskData[setType][sample].shape for sample in sampleFileTypes])

print("Reshaping...")
# Arr dimensions: N_events x N_SV(10) x SV_Vars(7). Want to combine N_Events and N_SV to get a 2D array of (N_events*N_SV) x SV_Vars
# reShapeMaskData = { "validation":{"W":None, "Z":None, "H":None, "t":None, "b":None, "QCD":None}, 
#                     "test":{"W":None, "Z":None, "H":None, "t":None, "b":None, "QCD":None}, 
#                     "train":{"W":None, "Z":None, "H":None, "t":None, "b":None, "QCD":None} }
for setType in setTypes:
    for sample in sampleFileTypes:
        # arr = maskData[setType][sample]
        maskData[setType][sample] = np.ravel(maskData[setType][sample])
        # reShapeMaskData[setType][sample] = arr.reshape( (arr.shape[0] * arr.shape[1] , arr.shape[2]) ))    
        # maskData[setType][sample] = arr
    # del data[setType]
# del data

for setType in setTypes: print("New " + setType + " mask shape: ", [maskData[setType][sample].shape for sample in sampleFileTypes])

for setType in setTypes: 
    for sample in sampleFileTypes:
        maskData[setType][sample] = maskData[setType][sample] != 0


for index, var in enumerate(allVars):
    # if (var == "px") or (var == "py") or (var == "pz"): continue
    if var != "pdgId": continue
    print("Beginning " + var)
    print("Loading h5 var data... ")
    # h5Key = "PF_cands_LabFrame"
    data = {}
    for setType in setTypes:
        print("Loading " + setType)
        data[setType] = {}
        for sample in sampleFileTypes:
            # data[setType][sample] = np.array(h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + setType + "_flattened.h5","r")[h5Key][:,:,index])
            data[setType][sample] = np.array(h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + setType + ".h5","r")[h5Key][:,:,index])

    
    # print("Loading post scale h5py files")
    # postScaleEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + setType + "_flattened_standardized.h5","r")["BES_vars"])[:,myMask] for mySample in sampleFileTypes]
    for setType in setTypes: print("Initial " + setType + " events shape: ", [data[setType][sample].shape for sample in sampleFileTypes])

    print("Reshaping...")
    for setType in setTypes:
        for sample in sampleFileTypes:
            arr = data[setType][sample]
            data[setType][sample] = np.ravel(data[setType][sample])
    for setType in setTypes: print("New " + setType + " events shape: ", [data[setType][sample].shape for sample in sampleFileTypes])

    for setType in setTypes:
        print(setType)
        countDict = {}
        allVals = []

        for sample in sampleFileTypes:
            # print(sample)
            countDict[sample] = {} 
            values, counts = np.unique(data[setType][sample], return_counts=True)
            for i, value in enumerate(values):
                val = int(value)
                if val not in allVals: allVals.append(val)
                countDict[sample][val] = counts[i]

        allVals.sort()
        for sample, d in countDict.items():
            for value in allVals:
                if value not in d: countDict[sample][value] = 0

        print("pdgids: " + str(allVals) )
        for sample in sampleFileTypes:
            print(sample + ": " + str([countDict[sample][value] for value in allVals]) )

    print("Trimming...")
    for setType in setTypes:
        for sample in sampleFileTypes:
            arr      = data[setType][sample]
            arr_mask = maskData[setType][sample]
            data[setType][sample] = arr[arr_mask]    
            
    for setType in setTypes: print("Trimmed " + setType + " events shape: ", [data[setType][sample].shape for sample in sampleFileTypes])

    for setType in setTypes:
        print(setType)
        countDict = {}
        allVals = []

        for sample in sampleFileTypes:
            # print(sample)
            countDict[sample] = {} 
            values, counts = np.unique(data[setType][sample], return_counts=True)
            for i, value in enumerate(values):
                val = int(value)
                if val not in allVals: allVals.append(val)
                countDict[sample][val] = counts[i]

        allVals.sort()
        for sample, d in countDict.items():
            for value in allVals:
                if value not in d: countDict[sample][value] = 0

        print("pdgids: " + str(allVals) )
        for sample in sampleFileTypes:
            print(sample + ": " + str([countDict[sample][value] for value in allVals]) )

    quit()
    print("Plotting...")
    # plotDir = "plots/SVvars/" + setType + "/"
    # plotDir = "plots/vars/SVvars/"
    plotDir = "plots/vars/PF_AllFrame/"
    plt.figure()
    saveDir = plotDir + var + "/"
    if not os.path.isdir(saveDir): os.makedirs(saveDir)
    # for sample, setArrays in data.items():
    # for i, sample in enumerate(sampleFileTypes):
        # print("Plotting " + sample)
    for setType in setTypes:
        print("Plotting " + setType)
        # saveDir = plotDir + var + "_" + setType + "/"

        maxVal = 0
        minVal = 0
        # if ("Track" in var) or ("dof" in var):
        # for arr in data[sample].values():
        # for setType in setTypes:
            # print(setType)
        for sample in sampleFileTypes:
            arr = data[setType][sample]
            # arr = data[setType][sample][0]
            # mintemp = int(np.amin(arr))
            # maxtemp = int(np.amax(arr))
            mintemp = np.amin(arr)
            maxtemp = np.amax(arr)
            # print("Temp", mintemp, maxtemp)
            if mintemp < minVal: minVal = mintemp
            if maxtemp > maxVal: maxVal = maxtemp
            # print("Val", minVal, maxVal)
        if var != "PUPPI_Weights" and var != "charge":
            maxVal = int(maxVal)
            minVal = int(minVal)
        # if ("Track" in var) or ("dof" in var): myBins = maxVal
        # elif "phi" in var:                     myBins = 21
        # else:                                  myBins = 51
        myBins = 51

        # --- Create Pre Scale histogram, legend and title ---
        title = var + "_" + setType  
        # title = var + "_" + samples[i]  
        # for setType, arr in data[sample].items():
        # for setType in setTypes:
        for sample in sampleFileTypes:
            arr = data[setType][sample]
            # arr = data[setType][sample][0]
            # plt.hist(arr[0], bins=myBins, range = (minVal, maxVal), histtype='step')
            plt.hist(arr, bins=myBins, range = (minVal, maxVal), histtype='step')
            # plt.hist(arr, bins=myBins, histtype='step')

            # if ("Track" in var) or ("dof" in var): 
            #     plt.hist(array[:,index], bins=maxVal, range = (minVal, maxVal), histtype='step', normed = True)
            # else: 
            #     plt.hist(array[:,index], bins=51, range = (minVal, maxVal), histtype='step', normed = True)
            # Check for bad values
            # minVal = np.amin(array[:,index])
            # if minVal == -999.99: print("BAD VALUE: ", title, samples[i])
        # plt.legend(frameon=False, labels = setTypes)
        plt.legend(frameon=False, labels = samples)
        plt.title( title )
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.yscale('log')
        plt.title( title + "_LogScale")
        plt.show()
        plt.savefig(saveDir + title + "_log.png")
        plt.clf()



        # --- Create Pre Scale histogram, legend and title ---
        # title = var + "_" + samples[i] + "_Normed" 
        title = var + "_" + setType + "_Normed" 
        # for setType, arr in data[sample].items():
        # for setType in setTypes:
        for sample in sampleFileTypes:
            # arr = data[setType][sample][0]
            arr = data[setType][sample]
            # plt.hist(arr[0], bins=myBins, range = (minVal, maxVal), histtype='step', normed = True)
            plt.hist(arr, bins=myBins, range = (minVal, maxVal), histtype='step', normed = True)
            # plt.hist(arr, bins=myBins, histtype='step', normed = True)
            # if ("Track" in var) or ("dof" in var): 
            #     plt.hist(array[:,index], bins=maxVal, range = (minVal, maxVal), histtype='step', normed = True)
            # else: 
            #     plt.hist(array[:,index], bins=51, range = (minVal, maxVal), histtype='step', normed = True)
            # Check for bad values
            # minVal = np.amin(array[:,index])
            # if minVal == -999.99: print("BAD VALUE: ", title, samples[i])
        # plt.legend(frameon=False, labels = setTypes)
        plt.legend(frameon=False, labels = samples)
        plt.title( title )
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.yscale('log')
        plt.title( title + "_LogScale")
        plt.show()
        plt.savefig(saveDir + title + "_log.png")
        plt.clf()
        

    plt.close()
"""

# This code plots SV/PF vars for train/val/test separately:
"""
    sampleFileTypes = ["WW","ZZ","HH","TT","BB","QCD"]
    samples     = ["W","Z","H","t","b","QCD"]
    h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
    setTypes = ["validation","test","train"]
    for setType in setTypes:
        print("Beginning " + setType)
        print("Loading h5py files")
        data  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + setType + "_flattened.h5","r")["SV_vars"])[()] for mySample in sampleFileTypes]

        # print("Loading post scale h5py files")
        # postScaleEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + setType + "_flattened_standardized.h5","r")["BES_vars"])[:,myMask] for mySample in sampleFileTypes]

        print("Initial events shape:", [arr.shape for arr in data])

        print("Reshaping...")
        # Arr dimensions: N_events x N_SV(10) x SV_Vars(7). Want to combine N_Events and N_SV to get a 2D array of (N_events*N_SV) x SV_Vars
        reshapedEvents = []
        # for arr in data: 
        #     oldShape = arr.shape
        #     reshapedEvents.append(arr.reshape( (oldShape[0]*oldShape[1], oldShape[2]) ) )
        for arr in data: arr = reshapedEvents.append(arr.reshape( (arr.shape[0] * arr.shape[1] , arr.shape[2]) ) )    
        del data
        print("New events shape:", [arr.shape for arr in reshapedEvents])

        print("Trimming...")
        trimmedEvents = []
        # 10 SV's is the max--if an event has less than 10, the SV_vars are not filled. Trim these out by checking the SV_nTracks (index 4):
        for arr in reshapedEvents: trimmedEvents.append(arr[arr[:,4] != 0])
        # for arr in reshapedEvents: trimmedEvents.append(arr[arr[:,1] > 0.1])
        del reshapedEvents
        data = trimmedEvents
        del trimmedEvents
        print("Trimmed events shape:", [arr.shape for arr in data])

        
        print("Plotting...")
        # plotDir = "plots/SVvars/" + setType + "/"
        # plotDir = "plots/vars/SVvars/"
        plotDir = "plots/vars/PF_LabFrame/"
        plt.figure()
        for index, var in enumerate(allVars):
            # print("Plotting " + var)
            # saveDir = plotDir + var + "_" + setType + "/"
            saveDir = plotDir + var + "/"
            if not os.path.isdir(saveDir): os.makedirs(saveDir)

            maxVal = 0
            minVal = 0
            # if ("Track" in var) or ("dof" in var):
            for arr in data:
                mintemp = int(np.amin(arr[:,index]))
                maxtemp = int(np.amax(arr[:,index]))
                if mintemp < minVal: minVal = mintemp
                if maxtemp > maxVal: maxVal = maxtemp

            if ("Track" in var) or ("dof" in var): myBins = maxVal
            elif "phi" in var:                     myBins = 21
            else:                                  myBins = 51

            # --- Create Pre Scale histogram, legend and title ---
            title = var + "_" + setType  
            for i, array in enumerate(data):
                plt.hist(array[:,index], bins=myBins, range = (minVal, maxVal), histtype='step', normed = True)
                # if ("Track" in var) or ("dof" in var): 
                #     plt.hist(array[:,index], bins=maxVal, range = (minVal, maxVal), histtype='step', normed = True)
                # else: 
                #     plt.hist(array[:,index], bins=51, range = (minVal, maxVal), histtype='step', normed = True)
                # Check for bad values
                # minVal = np.amin(array[:,index])
                # if minVal == -999.99: print("BAD VALUE: ", title, samples[i])
            plt.legend(frameon=False, labels = samples)
            plt.yscale('log')
            plt.title( title )
            plt.show()
            plt.savefig(saveDir + title + "_logNorm.png")
            plt.clf()
            

        plt.close()    
    """

# This code plots the normal BES_vars:
#"""
# setTypes = ["test"]
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_longLearn/300Basic_300_Z/masspT.txt"
# mask, _ = tools.loadMask(maskPath, 596)
for year in years:
    print("Beginning " + year)
    for setType in setTypes:
        print("Beginning " + setType)
        print("Loading pre scale h5py files")
        # preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + setType + "_" + suffix + ".h5","r")["BES_vars"])[:,mask] for mySample in sampleFileTypes]
        # preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + setType + "_" + suffix + ".h5","r")["BES_vars"])[()] for mySample in sampleFileTypes]
        
        preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_" + setType +"_"+ suffix+ ".h5","r")["BES_vars"])[()] for mySample in sampleFileTypes]

        # preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_" + setType + "_" + suffix + "_standardized.h5","r")["BES_vars"])[()] for mySample in sampleFileTypes]

        """
        # print("Loading post scale h5py files")
        # postScaleEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + setType + "_flattened_standardized.h5","r")["BES_vars"])[:,myMask] for mySample in sampleFileTypes]

        # print("Pre scale events shape:", [arr.shape for arr in preScaleEvents])
        # print("Post scale events shape:",[arr.shape for arr in postScaleEvents])

        # print("Unscaling...")
        # scalePath = "/uscms/home/bonillaj/nobackup/Brendan/CMSSW_10_2_18/src/centralBEST/BEST/training/ScalerParameters_" + setType + ".txt"
        # scaleFile = open(scalePath, "r")
        # means  = []
        # scales = []
        # i = -1
        # for line in scaleFile:
        #     i += 1
        #     if str(i) not in varDict: continue
        #     mean, vari = line.split(',')
        #     vari = vari.strip()
        #     means.append(float(mean))
        #     # scales.append(float(vari))
        #     scales.append(float(vari) ** 2)
        # scaleFile.close()

        # scaler = preprocessing.StandardScaler()
        # scaler.mean_ = np.array(means)
        # scaler.scale_ = np.array(scales)
        # # scaler.var_ = np.array(scales)
        # unScaledEvents = [scaler.inverse_transform(scaledEvents) for scaledEvents in postScaleEvents]
        # print("Unscaled events shape:",[arr.shape for arr in unScaledEvents])
        # print("Plotting pT...")
        """

        # plotDir = "plots/BESvars/" + setType + "/"
        # plotDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/plots/vars/BESvars_OGStandardScaler/test/"


        plotDir = os.path.join("plots/BESvars/",setType)
        plt.figure()
        # for index, var in enumerate(allVars):
        for index in range(len(varDict.keys())):
            # print("Plotting " + var)
            var = varDict[str(index)]
            # saveDir = plotDir + var + "_" + setType + "/"
            # saveDir = os.path.join(plotDir, var)
            # compressDir = var
            if   "FoxWolf" in var: compressDir = "foxwolf"
            elif "aplanarity" in var: compressDir = "aplanarity"
            elif "asymmetry" in var: compressDir = "asymmetry"
            elif "sphericity" in var: compressDir = "sphericity"
            elif "thrust" in var: compressDir = "thrust"
            elif "DeltaCosTheta" in var: compressDir = "deltacos"
            elif "CosTheta" in var: compressDir = "cos"
            elif "_mass_" in var: compressDir = "mass"
            elif "jet_energy" in var: compressDir = "energy"
            elif "jet_px" in var: compressDir = "px"
            elif "jet_py" in var: compressDir = "py"
            elif "jet_pz" in var: compressDir = "pz"
            else: compressDir = "invariant"




            saveDir = os.path.join(plotDir, compressDir)
            if not os.path.isdir(saveDir): os.makedirs(saveDir)

            maxVal = 0
            minVal = np.inf
            # if ("Track" in var) or ("dof" in var):
            for arr in preScaleEvents:
                mintemp = int(np.amin(arr[:,index]))
                maxtemp = int(np.amax(arr[:,index]))
                if mintemp < minVal: minVal = mintemp
                if maxtemp > maxVal: maxVal = maxtemp

            # minVal = 0
            # if   var == "jetAK8_mass": maxVal = 300
            # elif var == "jetAK8_SoftDropMass": maxVal = 225
            # --- Create Pre Scale histogram, legend and title ---
            # title = var + "_" + year + "_" + setType + "_" + suffix + "_postScale" 
            # title = var + "_" + setType 
            title = var + "_" + year
            for i, array in enumerate(preScaleEvents):
                plt.hist(array[:,index], bins=51, range=(minVal, maxVal), histtype='step')
                # plt.hist(array[:,index], bins=51, histtype='step')
                # Check for bad values
                # minVal = np.amin(array[:,index])
                # if minVal == -999.99: print("BAD VALUE: ", title, samples[i])
            plt.legend(frameon=False, labels = samples)
            plt.title( title )
            plt.show()
            # plt.savefig(saveDir + title + ".png")
            # plt.savefig(os.path.join(saveDir,title + ".png"))
            plt.savefig(os.path.join(saveDir,title + ".pdf"))
            plt.clf()
            
            """
            # # --- Create Post Scale histogram, legend and title ---
            # title = var + "_" + setType + "_postScale" 
            # for i, array in enumerate(postScaleEvents):
            #     plt.hist(array[:,index], bins=51, histtype='step')
            #     # Check for bad values
            #     minVal = np.amin(array[:,index])
            #     if minVal == -999.99: print("BAD VALUE: ", title, samples[i])
            # plt.legend(frameon=False, labels = samples)
            # plt.title( title )
            # plt.show()
            # plt.savefig(saveDir + title + ".png")
            # plt.clf()

            # # --- Create Unscaled histogram, legend and title ---
            # title = var + "_" + setType + "_unscaled" 
            # for i, array in enumerate(unScaledEvents):
            #     plt.hist(array[:,index], bins=51, histtype='step')
            #     # Check for bad values
            #     minVal = np.amin(array[:,index])
            #     if minVal == -999.99: print("BAD VALUE: ", title, samples[i])
            # plt.legend(frameon=False, labels = samples)
            # plt.title( title )
            # plt.show()
            # plt.savefig(saveDir + title + ".png")
            # plt.clf()
            """
        plt.close()
    # """


# Check how long the script took to run
tools.logTime(startTime)

