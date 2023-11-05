#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# functions.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# This module contains functions to be used while training BEST ///////////////////
#==================================================================================

# modules
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import numpy.random
import itertools
import os
import time
import datetime
import sys

# functions from modules
from shutil import rmtree
from scipy import interp
from sklearn import metrics, preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.externals.joblib import load

#==================================================================================
# Log Time ////////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Records time script takes to complete ///////////////////////////////////////////
#----------------------------------------------------------------------------------

def logTime(startTime=None, name=sys.argv[0]):
    if not startTime: return time.time()
    
    timeTaken = datetime.timedelta(seconds=int(time.time() - startTime))
    timeMessage = ("\n"+str(name)+" took " + str( timeTaken ) + " to complete.")
    print(timeMessage)
    with open("Logs/timeLog.txt", "a") as f:
        f.write(timeMessage)

#==================================================================================
# Plot Confusion Matrix ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# cm is the comfusion matrix //////////////////////////////////////////////////////
# classes are the names of the classes that the classifier distributes among //////
#----------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting `normalize=True`.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
       print("Normalized confusion matrix")
       title = "Normalized Confusion Matrix"
   else:
       print('Confusion matrix, without normalization')
       title = "Confusion Matrix"

   print(cm)

   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = numpy.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)

   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.tight_layout() #make all the axis labels not get cutoff

def plot_confusion_matrix_sam(cm, classes, plotDir, suffix,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("Plot confusion matrix")
    if not os.path.isdir(plotDir): os.makedirs(plotDir)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix, normalized")
        title = "Normalized Confusion Matrix"
        saveFile = plotDir + 'ConfusionMatrix_BES' + suffix + '_normalized.png'
    else:
        print('Confusion matrix, without normalization')
        title = "Confusion Matrix"
        saveFile = plotDir + 'ConfusionMatrix_BES' + suffix + '.png'

    print(cm)
    title = suffix + " " + title
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout() #make all the axis labels not get cutoff

    print("Saving to: " + saveFile)
    plt.savefig(saveFile)

    plt.clf()
    plt.close()

#==================================================================================
# Plot Performance ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# loss is an array of loss and loss_val from the training /////////////////////////
# acc is an array of acc and acc_val from the training ////////////////////////////
#----------------------------------------------------------------------------------

def plotPerformance(lossList, accList, suffix, plotDir): 
    loss = lossList[0]
    val_loss = lossList[1]
    acc = accList[0]
    val_acc = accList[1]

    # plot loss vs epoch
    plt.figure()
    plt.plot(loss, label='loss; Min loss: ' + str(np.min(loss))[:6] + ', Epoch: ' + str(np.argmin(loss)) )
    plt.plot(val_loss, label='val_loss; Min val_loss: ' + str(np.min(val_loss))[:6] + ', Epoch: ' + str(np.argmin(val_loss)) )
    plt.title(suffix + " loss and val_loss vs. epochs")
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if not os.path.isdir(plotDir): os.makedirs(plotDir)
    # plt.savefig(plotDir+suffix+"_loss.pdf")
    plt.savefig(plotDir+suffix+"_loss.png")
    plt.close()

    # plot accuracy vs epoch
    plt.figure()
    plt.plot(acc,     label='acc; Max acc: '  + str(np.max(acc))[:6] + ', Epoch: ' + str(np.argmax(acc)) )
    plt.plot(val_acc, label='val_acc; Max val_acc: ' + str(np.max(val_acc))[:6] + ', Epoch: ' + str(np.argmax(val_acc)) )
    plt.title(suffix + " acc and val_acc vs. epochs")
    plt.legend(loc="lower right")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.savefig(plotDir+suffix+"_acc.pdf")
    plt.savefig(plotDir+suffix+"_acc.png")
    plt.close()

#==================================================================================
# Plot Probabilities //////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# probs is an array of probabilites, labels, and colors ///////////////////////////
#    [ [probArray, label, color], .. ] ////////////////////////////////////////////
#----------------------------------------------------------------------------------

def plotProbabilities(plotDir, eventPredictions, truthTest, targetNames):

    print("Plotting Classification Probabilities")
    saveDir = plotDir + "classification_probs/"
    if not os.path.isdir(saveDir): os.makedirs(saveDir)
    
    plt.figure()
    for i, target in enumerate(targetNames):

        # --- Create Class. Prob. histogram, legend and title ---
        title = "Probability of " + target + " Classification" 
        for j, mylabel in enumerate(targetNames):
            tempMask = [True if j == k else False for k in truthTest]
            plt.hist(eventPredictions[tempMask,i], label = mylabel, bins = 20, range = (0,1), histtype='step', log = True)
            # normed = True ???
        plt.xlim(0.0,1.0)
        leg = plt.legend(ncol = 6, loc = 'upper center', bbox_to_anchor = (0.0,1.1,1.0,0.1), borderpad = 0.9, borderaxespad = 2.0 )
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(1.1)
        plt.xlabel( title )
        plt.gca().tick_params(axis = 'y', which = 'both', direction = 'in', left = True, right = True)
        plt.gca().tick_params(axis = 'x', direction = 'in', top = True, bottom = True)
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.clf()
    plt.close()

#==================================================================================
# Load Mask ///////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Loads Mask text file and returns Boolean masking array //////////////////////////
#----------------------------------------------------------------------------------

def loadMask(maskPath, max = 82):
    print("Loading mask: " + maskPath)
    maskIndex = []
    with open(maskPath, "r") as f:
        for line in f:
            maskIndex.append(line.split(':')[0])
    print("Mask size: " + str(len(maskIndex)))
    myMask = [True if str(ind) in maskIndex else False for ind in range(max)]
    # myMask = [True if str(ind) in maskIndex else False for ind in range(82)]
    # myMask = [True if str(ind) in maskIndex else False for ind in range(82)]

    return myMask

#==================================================================================
# Load h5 Data ////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Loads h5 data files for specified setTypes. Returns dictionary containing ///////
#  concatenated data arrays and concatenated truth arrays.      ///////////////////
#----------------------------------------------------------------------------------

def loadH5Data(h5Dir, mask, sampleTypes, setTypes, scale):
    if not scale == "": scale = "_" + scale

    dataDict = {}
    for mySet in setTypes:
        print("Loading h5 files for " + mySet)
        h5Path = "Sample_2018_BESTinputs_" + mySet + scale + ".h5" # This just makes the next few lines a bit more readable

        # Check if loading all variables. Quicker to NOT use mask in this case
        if np.all(mask): eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[()]     for mySample in sampleTypes]
        else:            eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[:,mask] for mySample in sampleTypes]

        print("My " + mySet + " events shape:", [eventArrays[i].shape for i in range(len(eventArrays))])

        # Create Truth arrays; shape: N_events x 6
        truthArrays = [np.zeros( (len(eventArrays[i]), len(sampleTypes)) ) for i in range(len(eventArrays))] 

        # Arrays are filled with zeros. Now set 1's to record Truth particle info
        for i in range(len(sampleTypes)):
            truthArrays[i][:,i] = 1.

        print("My " + mySet + " truth shape: ", [ truthArrays[i].shape for i in range(len(truthArrays)) ] )
        
        print("Concatenating...")
        dataDict[mySet + "Events"] = np.concatenate(eventArrays)
        dataDict[mySet + "Truth"]  = np.concatenate(truthArrays)
        del eventArrays
        del truthArrays

        print("My " + mySet + " concatenated event shape: ", dataDict[mySet + "Events"].shape)
        print("My " + mySet + " concatenated truth shape: ", dataDict[mySet + "Truth"].shape)

    print("Labels are: ", [ [i,mySample] for i,mySample in enumerate(sampleTypes)])
    print("Keys: ", dataDict.keys())
    print("Finished loading h5 data.")

    return dataDict

#==================================================================================
# Shuffler ///////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Shuffles dictionary of data arrays.  ////////////////////////////////////////////
#----------------------------------------------------------------------------------

def shuffleArray(arrayDict, rng_state=np.random.get_state()):
    for key, array in arrayDict.items():
        if "test" in key: continue
        print("Shuffling " + key)
        np.random.set_state(rng_state)
        np.random.shuffle(array)

#==================================================================================
# Load Scaler Model ///////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Loads full scaler model, returns masked scaler model.  //////////////////////////
#   This function is depreciated; it is intended to be used when one has a scaler /
#   model, but has not saved the scaled data to an h5 file. In this case, the /////
#   input data will need to be scaled on the fly before being given to the NN. ////
#   This function will create a scaler object that matches the data that needs ////
#   to be scaled. /////////////////////////////////////////////////////////////////
#   It is reccomended to instead save the scaled data in the h5 format. ///////////
#----------------------------------------------------------------------------------

def loadScalerModel(scale, mask):

    scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    scalePath = scaleDir + "ScalerParameters_" + scale + ".joblib"

    if not os.path.isfile(scalePath):
        print(scalePath, "does not exist")
        quit()
        
    print("Loading Scaler Model: " + scalePath)
    fullScaler = load(scalePath)

    # If loading all BES vars, simply return the full Scaler model
    if np.all(mask): return fullScaler

    # Select correct scaler model, then create new model with only relevant vars
    print("Applying Mask...")
    if "Quantile" in scalePath:
        if "Normal"  in scalePath: scaler = preprocessing.QuantileTransformer(output_distribution = "normal")
        if "Uniform" in scalePath: scaler = preprocessing.QuantileTransformer(output_distribution = "uniform")
        scaler.quantiles_  = fullScaler.quantiles_[:,mask]
        scaler.references_ = fullScaler.references_[:]
    else:
        if "Standard" in scalePath:
            scaler = preprocessing.StandardScaler()
            scaler.mean_ = fullScaler.mean_[mask]
        elif "Min" in scalePath:
            if "01" in scalePath: scaler = preprocessing.MinMaxScaler(feature_range = (0,1))
            if "11" in scalePath: scaler = preprocessing.MinMaxScaler(feature_range = (-1,1))
            scaler.min_ = fullScaler.min_[mask]
        elif "Abs" in scalePath:
            scaler = preprocessing.MaxAbsScaler()
            scaler.max_abs_ = fullScaler.max_abs_[mask]
        else:
            print("Error: Invalid Scaler Model: " + scalePath)
            quit()
        scaler.scale_ = fullScaler.scale_[mask]
    del fullScaler

    return scaler

#==================================================================================
# Record Classification Rate //////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Record the rate at which each class is being tagged. ////////////////////////////
#   Useful for checking if the NN is biased toward a certain class. ///////////////
#   Provides the same information as the Confusion Matrix. ////////////////////////
#----------------------------------------------------------------------------------

def recordClassification(cm, truthTest, targetNames, modelType, suffix):

    print("Recording Classifcation Rates...")
    logFile = "Logs/" + modelType + "_classifyLog.txt"
    classifylog = open(logFile, "a") 
    classifylog.write("-----------------------------------\n")
    classifylog.write("Running " + suffix + ":\n")

    totalTested = np.bincount(truthTest)
    classifylog.write("\t\t\t\t\t\t\t\t\t\tW\t Z\t  H\t  Top\tb  QCD  Total\n")
    for i, target in enumerate(targetNames):
        totalPredicted = cm[:,i]
        classifyMessage = "\t" + target + " Category: Tagger predicted\t" + str( totalPredicted ) + " " + str(np.sum(totalPredicted)) + " out of " + str(totalTested[i]) + " (" + str(100 * (float(totalPredicted[i])/float(totalTested[i]))  )[:6] + "%) truth events.\n"        
        print(classifyMessage)
        classifylog.write(classifyMessage)

    classifylog.write("-----------------------------------\n")
    classifylog.close
    print("Find Classification Rates at: " + logFile)

#==================================================================================
# Make Strings //////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
#  ////////////////////////////

#----------------------------------------------------------------------------------

# def makeStrings(modelType, maskPath, scale, year, outDir, suffix, replaceModel=True):
def makeStrings(modelType, args, maskPath, suff, replaceModel=True):

    # scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    # scalePath = scaleDir + "ScalerParameters_" + scale + ".joblib"

    # maskName  = args.maskPath[1 + args.maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask
    maskName  = maskPath[1 + maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask

    # mySuffix = suffix + scale + year + "_" + maskName[:-4] + "_" + modelType
    # mySuffix = args.suffix + args.scale[8:] + maskName[11:-4] 
    # mySuffix = suff + args.scale[8:] + maskName[11:-4] 
    mySuffix = suff + args.scale[8:] + maskName[:-4] 
    # mySuffix = suffix + "_" + scale[8:]

    plotDir  = "plots/" + modelType + "/" + mySuffix + "/"
    modelDir = args.outDir + modelType + "/" + mySuffix + "/"
    modelFile = modelDir + "BEST_model_" + mySuffix + ".h5"
    maskSave  = modelDir + maskName    
    # historyFile = modelDir + "history_" + mySuffix + ".joblib"
    if not os.path.isdir(plotDir):  os.makedirs(plotDir)

    if replaceModel:
        print("Creating Directories....")
        if os.path.isdir(modelDir): 
            print("Deleting old model directory...")
            rmtree(modelDir)
        print("Creating new model directory at: \n" + modelDir)
        os.makedirs(modelDir)
        # os.system('cp ' + args.maskPath + ' ' + maskSave)
        os.system('cp ' + maskPath + ' ' + maskSave)

    print(modelFile)
    print(mySuffix)

    return modelDir, plotDir, mySuffix

#==================================================================================
# Create ROC Curves ///////////////////////////////////////////////////////////////
#==================================================================================

def plotROC(BESpredict, truthLabels, plotDir, modelType, suffix):

    print("Plotting ROC curves")
    samples = ["Zt","Ht","Wb","QCD1500to2000","QCD2000toInf","TTBar"]
    saveDir = plotDir + "roc/" 
    if not os.path.isdir(saveDir): os.makedirs(saveDir)

    # Compute ROC curve and area for each class
    n_classes = truthLabels.shape[1] 
    fprBES = dict()
    tprBES = dict()
    roc_auc_BES = dict()
    # for i in range(n_classes):
    for i, sample in enumerate(samples):
        fprBES[sample], tprBES[sample], _ = roc_curve(truthLabels[:, i], BESpredict[:, i])
        roc_auc_BES[sample] = auc(fprBES[sample], tprBES[sample])

    # Compute micro-average ROC curve and ROC area
    fprBES["micro"], tprBES["micro"], _ = roc_curve(truthLabels.ravel(), BESpredict.ravel() )
    roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

    # Compute macro-average ROC curve and ROC area:

    # First aggregate all false positive rates
    all_fprBES = numpy.unique(numpy.concatenate([fprBES[sample] for sample in samples]) )

    # Interpolate all roc curves
    mean_tprBES = numpy.zeros_like(all_fprBES)
    for sample in samples:
        mean_tprBES += interp(all_fprBES, fprBES[sample], tprBES[sample] )

    # Average and compute macro AUC
    mean_tprBES /= n_classes

    fprBES["macro"] = all_fprBES
    tprBES["macro"] = mean_tprBES
    roc_auc_BES["macro"] = auc(fprBES["macro"], tprBES["macro"])

    # Fill dictionary with plot label information
    labelDict = {} # { key: [plot label, plot title, plot path name], ... }
    for key in roc_auc_BES.keys():
        if   key == "micro": labelDict[key] = ["Micro Average", "average_micro"]
        elif key == "macro": labelDict[key] = ["Macro Average", "average_macro"]
        else:                labelDict[key] = [key + " Category", key]

    print("Macro Average ROC AUC: " + str(roc_auc_BES["macro"]))
    print("Micro Average ROC AUC: " + str(roc_auc_BES["micro"]))

    # Record ROC AUC
    rocLog = open("Logs/" + modelType + "_rocLog.txt", "a") 
    # rocLog.write("AOCs for " + suffix + ":\nAvg: " + str(roc_auc_BES["macro"])[:8] + ", ")
    # rocLog.write("AUCs for " + suffix + ": Avg: " + str(roc_auc_BES["macro"])[:8] + ", ")

    #could put a bunch of spaces here before avg, length 30 - len(suffix)?
    spaces = " "*(30 - len(suffix))
    rocLog.write(suffix + ":" + spaces + "Avg: " + str(roc_auc_BES["macro"])[:8] + ", ")
    for sample in samples: 
        rocLog.write(sample + ": " + str(roc_auc_BES[sample])[:8] + ", ")
    rocLog.write("avg: " + str(roc_auc_BES["micro"])[:8] + "\n")
    rocLog.close

    # Plot ROC Curves
    for key, rocAUC in roc_auc_BES.items():

        # Assign these for readability:
        title = labelDict[key][0] + " ROC Curve" 
        path  = saveDir + labelDict[key][1] + '_ROCplot'

        title = suffix + " " + title

        plt.figure(1)
        plt.plot(fprBES[key], tprBES[key],
                # label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                label= 'BES only ROC Curve (area = ' + str(rocAUC)[:6] + ') ',
                color='orange', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig(path + '.png')
        plt.close()

        # plt.figure(1)
        # plt.plot(fprBES[key], tprBES[key],
        #         label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
        #         color='orange', linewidth=2)

        # plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlim([0.0, 1.0])
        # # plt.ylim([0.0, 1.05])
        # plt.yscale("log")
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title(title + " Log Scale")
        # plt.legend(loc="lower right")
        # plt.savefig(path + '_log.png')
        # plt.close()

def dirStrings(args):
    # Create useful strings
    maskName = args.maskPath[1 + args.maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask
    mySuffix = args.suffix + args.scale + maskName[11:-4] 

    #plotDir  = "plots/" + args.modelType + "/" + mySuffix + "/"
    #modelDir = args.outDir + args.modelType + "/" + "2018_BESvarList_tweakedOldBEST" + "/"
    plotDir = "plots/tweakedOldBEST/"
    modelDir = "/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/training/trainingOutput/tweakedOldBEST/2018_BESvarList_tweakedOldBEST/"
    print("plotDir/modelDir:",plotDir, "/", modelDir)     
    #modelFile = modelDir + "BEST_model_" + mySuffix + ".h5"    
    #modelFile = modelDir + "BEST_model_" + mySuffix + ".pb"  #THIS IS WHAT WAS ORIGINALLY USED
    modelFile = "/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/training/trainingOutput/tweakedOldBEST/2018_BESvarList_tweakedOldBEST/BEST_model_2018_BESvarList_tweakedOldBEST.h5"
    maskSave  = modelDir + maskName    
    historyFile  = modelDir + "history_" + mySuffix + ".joblib" 

    # Make/check directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()

    if args.train: # Train model and plot performance
        if os.path.isfile(modelDir): 
            print("modelDir: ", modelDir)
            if not args.replace:
                print("Error, modelDir already exists. To replace model and plots, add -r flag when running script.")
                quit()
            
            print("Replacing directories...")
            rmtree(modelDir)
            if os.path.isdir(plotDir): rmtree(plotDir)

        os.makedirs(modelDir)
        os.makedirs(plotDir)
        os.system('cp ' + args.maskPath + ' ' + maskSave)
        
    else: # Plot performance of an already trained model
        if not os.path.isfile(modelFile): 
                print(modelFile, "does not exist")
                quit()
        if os.path.isdir(plotDir):
            print("plotDir", plotDir)
            if not args.replace:
                print("Error, plotDir already exists. To replace plots, add -r flag in addition to the -t flag.")
                quit()
            print("Replacing plot directory...")
            rmtree(plotDir)
        os.makedirs(plotDir)
    
    print("Suffix is: ", mySuffix)
    
    # Pack strings into dictionary for compact readability
    strings = { "modelFile":modelFile, "historyFile":historyFile, 
                "plotDir":plotDir, "suffix":mySuffix } 
    return strings

def plotpTCM(BESpredict, truthLabels, pTArray, plotDir, suffix):

    print("Plotting pT Performance")
    saveDir = plotDir + "tagging/" 
    if not os.path.isdir(saveDir): os.makedirs(saveDir)

    cm = {}
    # binsize = 25
    # bins_list = [i*binsize for i in range(20,64)]
    binsize = 200
    # bins_list = [i*binsize for i in range(0,60)]
    # suffix = suffix + "_mass"
    bins_list = [(i*binsize+1200.) for i in range(0,10)]
    suffix = suffix + "_SoftDrop"
    #print(np.min(pTArray),np.max(pTArray))


    for pTbin in bins_list:
        # Select events within certain pT range, create CM, save to dictionary
        ptIndex =  np.where(np.logical_and(pTArray >= pTbin, pTArray < (pTbin + binsize))) 
        #print(truthLabels.shape, BESpredict.shape, len(ptIndex), ptIndex)
        cmTemp = metrics.confusion_matrix(truthLabels[ptIndex], np.argmax(BESpredict[ptIndex], axis=1), labels=[0,1,2,3,4,5])
        #cmTemp = np.nan_to_num(cmTemp, nan = 0.001)
	# Normalize
        cm[pTbin] = cmTemp.astype('float') / cmTemp.sum(axis=1)[:, np.newaxis]
        
    targetNames = ["Zt","Ht","Wb","QCD1500to2000","QCD2000toInf","TTBar"]
    for i, target in enumerate(targetNames):
        myPtArrays = [cm[pTbin][:,i] for pTbin in bins_list]
        # print(len(bins_list), len(myPtArrays))
        # print(bins_list, myPtArrays)

        # --- Create histogram, legend and title ---
        plt.figure()
        plt.plot(bins_list, myPtArrays)
        plt.legend(targetNames, title = "True Particle")
        plt.title("Tagging Rate for " + target )
        # plt.xlabel("Jet pT (GeV)")
        # plt.xlabel("Jet Mass (GeV)")
        plt.xlabel("Event HT")
        plt.ylabel("Tagged Rate")
        plt.show()
        plt.savefig(saveDir + suffix + '_tagRate_' + target + '.png')
        plt.clf()

        myPtArrays = [cm[pTbin][i,:] for pTbin in bins_list]
        plt.figure()
        plt.plot(bins_list, myPtArrays)
        plt.legend(targetNames, title = "Tagged As")
        plt.title("Tagging Efficiency for " + target )
        # plt.xlabel("Jet pT (GeV)")
        # plt.xlabel("Jet Mass (GeV)")
        plt.xlabel("Event HT")
        plt.ylabel("Tagging Efficiency")
        plt.show()
        plt.savefig(saveDir + suffix + '_efficiency_' + target + '.png')
        plt.clf()

    print("Finished, check out your new plots at:")
    print(plotDir)
def plotAll(args, strings, truthData, modelType, BESpredict, year):
    print("Plotting BEST Performance")

    # Unpack strings for readability
    modelFile = strings["modelFile"] 
    historyFile = strings["historyFile"] 
    plotDir = strings["plotDir"] 
    suffix = strings["suffix"] 

    # plotDir = os.path.join(plotDir,testSet+'/')
    if not os.path.isdir(plotDir): os.makedirs(plotDir)

    # scalePath = "ScalerParameters_" + args.suffix + "/BESTScalerParameters_" + year + ".joblib"    
    # scalePath = "ScalerParameters_" + args.suffix + "/BESTScalerParameters_" + year + ".txt"    
    # mask, varDict = loadMask(args.maskPath)    
    # r_varDict = {v:k for k,v in varDict.items()} #invert dictionary 

    # Accuracy and Loss plots
    plotAccLoss(historyFile, suffix, plotDir)

    samples = ['Signal','QCD', 'TTBar']
    # sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
    # sampleTypes = ["WW","ZZ","HH","ZPTT","BB","QCD"]

    # dataDict = loadH5Data(args, mask, sampleTypes, ["test"], {"test":None}, year, testSet)
    # scaler = loadScalerModel(args.suffix, year, mask)
    # dataDict["testEvents"] = scaler.transform(dataDict["testEvents"])
    # del scaler
    
    # if np.all(mask): eventData = dataDict["testEvents"]
    # else:            eventData = dataDict["testEvents"][:,mask]
    
    # eventData = manualScale(dataDict["testEvents"], scalePath, r_varDict)

    # # Load model
    # print("Using BEST to predict...")
    # model_BEST = load_model(modelFile)
    # # BESpredict  = model_BEST.predict(eventData)
    # BESpredict  = model_BEST.predict(dataDict["testEvents"])
    # del model_BEST

    # truthData = dataDict["testTruth"]
    # del dataDict

    # Plot ROC Curve
    # plotROC(BESpredict, truthData, plotDir, samples, modelType, suffix)
    plotROC(BESpredict, truthData, plotDir, samples, args.modelType, suffix)

    # Collapse truth labels into 1D (length N_events) array containing truth index (0-5) for each event 
    truthData = np.argmax(truthData, axis=1) 

    # Plot Classification Probabilities
    # plotProbabilities(plotDir, BESpredict, truthData, samples)
    plotProbabilities(plotDir, BESpredict, truthData, samples, year)
    
    print("Making CM")
    print("My predictions shape:",      BESpredict.shape)
    print("Corresponding truth shape:", truthData.shape)
    
    cm = metrics.confusion_matrix(truthData, np.argmax(BESpredict, axis=1) )
                           
    # Plot Confusion Matrix, both normalized and not normalized
    plot_confusion_matrix(cm, samples, plotDir, suffix)
    plot_confusion_matrix(cm, samples, plotDir, suffix, normalize=True)
    # plot_confusion_matrix(cm, samples, plotDir, suffix, normalize=True, compare=True, args=args, year=year, testSet=testSet)

    # Record classification rates 
    # recordClassification(cm, truthData, samples, modelType, suffix)
    recordClassification(cm, truthData, samples, args.modelType, suffix)

    # Plot Efficiency
    plotpTCM(BESpredict, truthData, plotDir, args, year)
    # plotpTCM(BESpredict, truthData, plotDir, args, year, testSet)
    # plotpTCM(scaledTestEvents, truthData, testDataDict["testEvents"][:,548], plotDir, suffix)


    print("Finished Plotting BEST Performance. Check plots out at:")
    print(plotDir)

# plotters
# make strings/dirs
# timelog

# Test plotters, make sure the results are the same now that they functions
# then test plotters for running on full stats, ensure they are the same
#then include ROC in the plotting script, rename
