#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# functions.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Sam Abbott, Reyer Band, Johan S. Bonilla, Brendan Regnary ////////////
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
import tensorflow as tf

# functions from modules
from shutil import rmtree
from scipy import interp
from sklearn import metrics, preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.externals.joblib import load

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import load_model

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

def plot_confusion_matrix(cm_in, classes, plotDir, suffix,
                          normalize=False,
                          compare=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,#): 
                          args=None, year='', testSet=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("Plot confusion matrix")
    saveDir = os.path.join(plotDir,"ConfusionMatrix/")
    if not os.path.isdir(saveDir): os.makedirs(saveDir)

    if compare:
        print("Confusion matrix, standard")
        title = "Standard Confusion Matrix"
        saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix + '_standard.png')
        standardCM = getStandardCM()

        print("Confusion matrix, compared")
        cm_temp = cm_in.astype('float') / cm_in.sum(axis=1)[:, np.newaxis]
        title = "Compared Confusion Matrix"
        saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix + '_compared.png')
        cm = np.subtract(cm_temp,standardCM)
    elif normalize:
        cm = cm_in.astype('float') / cm_in.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix, normalized")
        title = "Normalized Confusion Matrix"
        saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix + '_normalized.png')
    else:
        cm = cm_in
        print('Confusion matrix, without normalization')
        title = "Confusion Matrix"
        saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix + '.png')

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
    plt.savefig(saveFile, bbox_inches='tight')

    saveFile = saveFile[:-3] + 'pdf'
    print("Saving to: " + saveFile)
    plt.savefig(saveFile, bbox_inches='tight')
    
    plt.clf()
    plt.close()


def getStandardCM():
# def getStandardCM(args, year, testSet):
    cm = np.array([
        [0.71,0.18,0.01,0.03,0.02,0.05],
        [0.21,0.64,0.05,0.03,0.02,0.04],
        [0.02,0.09,0.74,0.04,0.08,0.03],
        [0.04,0.03,0.05,0.78,0.04,0.05],
        [0.02,0.02,0.05,0.05,0.68,0.17],
        [0.04,0.03,0.02,0.05,0.14,0.72]
    ])

    print(cm)

    return cm
    """
    scaleParamFile = "ScalerParameters_flattened/old/BESTParameters.txt" 
    mask, varDict = loadMask(args.maskPath)    
    r_varDict = {v:k for k,v in varDict.items()} #invert dictionary 

    sampleTypes = ["WW","ZZ","HH","ZPTT","BB","QCD"]
    # sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]

    dataDict = loadH5Data(args, mask, sampleTypes, ["test"], {"test":None}, year, testSet)

    # Collapse truth labels into 1D (length N_events) array containing truth index (0-5) for each event 
    truthData = np.argmax(dataDict["testTruth"], axis=1) 

    scaledData = manualScale(dataDict["testEvents"], scaleParamFile, r_varDict)
    
    # scaledData = np.zeros((len(truthData),len(mask)))
    # # Manually scale
    # with open(scaleParamFile, "r") as f:
    #     for line in f:
    #         # maskIndex.append(line.split(':')[0])
    #         var, scale, param1, param2 = line.split(',')
    #         param1 = float(param1)
    #         param2 = float(param2.strip())
    #         index = int(r_varDict[var])
    #         val = dataDict["testEvents"][...,index]
    #         if   scale == "NoScale":  result = val
    #         elif scale == "MinMax":   result = (val-param1)/(param2-param1)                
    #         elif scale == "Standard": result = (val-param2)/param1
    #         elif scale == "MaxAbs":   result = val/param1
    #         scaledData[...,index] = result

    del dataDict

    # Load model
    modelPath = "/uscms/home/msabbott/abbottBEST/training/models/recheck_long/140Basic300Wbothak8HT400/BEST_model_140Basic300Wbothak8HT400.h5"
    model = load_model(modelPath)
    BESpredict  = model.predict(scaledData)
    del model
    del scaledData

    cm = metrics.confusion_matrix(truthData, np.argmax(BESpredict, axis=1) )
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
    """

def manualScale(data, paramFile, r_varDict):
    print("Manually Scaling with: " + paramFile)
    scaledData = np.zeros((data.shape[0], data.shape[1]))
    # Manually scale
    with open(paramFile, "r") as f:
        for line in f:
            var, scale, param1, param2 = line.split(',')
            param1 = float(param1)
            param2 = float(param2.strip())
            index = int(r_varDict[var])
            val = data[...,index]
            if   scale == "NoScale":  result = val
            elif scale == "MinMax":   result = (val-param1)/(param2-param1)                
            elif scale == "Standard": result = (val-param2)/param1
            elif scale == "MaxAbs":   result = val/param1
            else: print("ERRRRORRRR")
            scaledData[...,index] = result    

    return scaledData

#==================================================================================
# Plot Performance ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# loss is an array of loss and loss_val from the training /////////////////////////
# acc is an array of acc and acc_val from the training ////////////////////////////
#----------------------------------------------------------------------------------

def plotAccLoss(historyFile, suffix, plotDir): 
    history = load(historyFile)

    # loss     = history.history["loss"]
    # val_loss = history.history["val_loss"]
    # acc      = history.history["acc"]
    # val_acc  = history.history["val_acc"]

    loss     = history["loss"]
    val_loss = history["val_loss"]
    acc      = history["acc"]
    val_acc  = history["val_acc"]

    # plot loss vs epoch
    plt.figure()
    plt.plot(loss, label='loss; Min loss: ' + str(np.min(loss))[:4] + ', Epoch: ' + str(np.argmin(loss)) )
    plt.plot(val_loss, label='val_loss; Min val_loss: ' + str(np.min(val_loss))[:4] + ', Epoch: ' + str(np.argmin(val_loss)) )
    plt.title(suffix + " loss and val_loss vs. epochs")
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if not os.path.isdir(plotDir): os.makedirs(plotDir)
    plt.savefig(plotDir+suffix+"_loss.pdf")
    plt.savefig(os.path.join(plotDir,suffix+"_loss.png"))
    plt.close()

    # plot accuracy vs epoch
    plt.figure()
    plt.plot(acc,     label='acc; Max acc: '  + str(np.max(acc))[:4] + ', Epoch: ' + str(np.argmax(acc)) )
    plt.plot(val_acc, label='val_acc; Max val_acc: ' + str(np.max(val_acc))[:4] + ', Epoch: ' + str(np.argmax(val_acc)) )
    plt.title(suffix + " acc and val_acc vs. epochs")
    plt.legend(loc="lower right")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(plotDir+suffix+"_acc.pdf")
    plt.savefig(os.path.join(plotDir,suffix+"_acc.png"))
    plt.close()

#==================================================================================
# Plot Probabilities //////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# probs is an array of probabilites, labels, and colors ///////////////////////////
#    [ [probArray, label, color], .. ] ////////////////////////////////////////////
#----------------------------------------------------------------------------------

# def plotProbabilities(plotDir, eventPredictions, truthTest, targetNames):
def plotProbabilities(plotDir, eventPredictions, truthTest, targetNames, year):

    print("Plotting Classification Probabilities")
    # saveDir = plotDir + "classification_probs/"
    saveDir = os.path.join(plotDir,"classification_probs/")
    if not os.path.isdir(saveDir): os.makedirs(saveDir)
    
    plt.figure()
    for i, target in enumerate(targetNames):

        # --- Create Class. Prob. histogram, legend and title ---
        title = "Probability of " + target + " Classification for " + year 
        for j, mylabel in enumerate(targetNames):
            tempMask = [True if j == k else False for k in truthTest]
            plt.hist(eventPredictions[tempMask,i], label = mylabel, bins = 20, range = (0,1), histtype='step', log = True)
        plt.xlim(0.0,1.0)
        leg = plt.legend(ncol = 4, loc = 'upper center', bbox_to_anchor = (0.0,1.1,1.0,0.1), borderpad = 0.9, borderaxespad = 2.0 )
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(1.1)
        plt.xlabel( title )
        plt.gca().tick_params(axis = 'y', which = 'both', direction = 'in', left = True, right = True)
        plt.gca().tick_params(axis = 'x', direction = 'in', top = True, bottom = True)
        plt.show()
        plt.savefig(saveDir + title + ".png")
        plt.savefig(saveDir + title + ".pdf")
        plt.clf()
    plt.close()

#==================================================================================
# Load Mask ///////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Loads Mask text file and returns Boolean masking array //////////////////////////
#----------------------------------------------------------------------------------

# def loadMask(maskPath, max = 551):
def loadMask(maskPath, max = 82):
    print("Loading mask: " + maskPath)
    maskIndex = []
    varDict = {}
    with open(maskPath, "r") as f:
        for line in f:
            # maskIndex.append(line.split(':')[0])
            index, var = line.split(':')
            maskIndex.append(index)
            varDict[index] = var.strip()    
    print("Mask size: " + str(len(maskIndex)))
    myMask = [True if str(ind) in maskIndex else False for ind in range(max)]
    # myMask = [True if str(ind) in maskIndex else False for ind in range(596)]
    # myMask = [True if str(ind) in maskIndex else False for ind in range(551)]

    return myMask, varDict


#==================================================================================
# Load h5 Data ////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Loads h5 data files for specified setTypes. Returns dictionary containing ///////
#  concatenated data arrays and concatenated truth arrays.      ///////////////////
#----------------------------------------------------------------------------------

def loadH5Data(args, mask, sampleTypes, setTypes, maxEvents, year='', testSet="flattened"):
    h5Dir = args.h5Dir
    scale = args.scale
    if year == '': year = args.year

    #if not scale == "": scale = "_" + scale

    dataDict = {}
    for mySet in setTypes:
        numEvents = maxEvents[mySet]

        print("Loading h5 files for " + mySet)
        h5Path = "Sample_"+year+"_BESTinputs_" + mySet + "_" + args.suffix + scale + ".h5" # This just makes the next few lines a bit more readable
        # if mySet == "test": h5Path = "Sample_"+year+"_BESTinputs_" + mySet + "_" + testSet + ".h5"
        print(h5Path)

        # Check if loading all variables. Quicker to NOT use mask in this case
        if np.all(mask): eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[:numEvents,:]     for mySample in sampleTypes]
        else:            eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[:numEvents,mask] for mySample in sampleTypes]

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

# def loadScalerModel(scale, mask):
def loadScalerModel(suffix, year, mask):

    # scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    # scalePath = scaleDir + "ScalerParameters_" + scale + ".joblib"
    scaleDir = "ScalerParameters_" + suffix + "/"
    scalePath = scaleDir + "BESTScalerParameters_" + year + ".joblib"

    # scalePath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/newBEST_Basic.joblib"
    if not os.path.isfile(scalePath):
        print(scalePath, "does not exist")
        quit()
        
    print("Loading Scaler Model: " + scalePath)
    # fullScaler = load(scalePath)
    fullScaler = load(scalePath, encoding='latin1')

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
        classifyMessage = "\t" + target + " Category: Tagger predicted\t" + str( totalPredicted ) + " " + str(np.sum(totalPredicted)) + " out of " + str(totalTested[i]) + " (" + str(100 * (float(totalPredicted[i])/float(totalTested[i]))  )[:4] + "%) truth events.\n"        
        print(classifyMessage)
        classifylog.write(classifyMessage)

    classifylog.write("-----------------------------------\n")
    classifylog.close
    print("Find Classification Rates at: " + logFile)

#==================================================================================
# Make dirs and strings //////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
#  Create and check the directories where the model and plots are saved. //////////
#   Also creates suffix string, which is unique to each model. ////////////////////
#----------------------------------------------------------------------------------

def dirStrings(args, year=''):
    # Create useful strings
    maskName = args.maskPath[1 + args.maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask
    # mySuffix = args.suffix + args.scale + maskName[11:-4] 
    if year == '': year = args.year
    mySuffix = args.suffix + "_" + year

    plotDir  = "plots/" + args.modelType + "/" + mySuffix + "/"
    modelDir = args.outDir + args.modelType + "/" + mySuffix + "/"
    
    modelFile = modelDir + "BEST_model_" + mySuffix + ".h5"
    # modelFile = modelDir + "BEST_model_" + mySuffix + ".pb"
    maskSave  = modelDir + maskName    
    historyFile  = modelDir + "history_" + mySuffix + ".joblib" 

    # Make/check directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()

    if args.train: # Train model and plot performance
        if os.path.isdir(modelDir): 
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
                "plotDir":plotDir, "suffix":mySuffix,
                "modelDir":modelDir } 
    return strings

#==================================================================================
# Create ROC Curves ///////////////////////////////////////////////////////////////
#==================================================================================

def plotROC(BESpredict, truthLabels, plotDir, samples, modelType, suffix):

    print("Plotting ROC curves")
    # samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]
    # saveDir = plotDir + "roc/" 
    saveDir = os.path.join(plotDir,"roc/")
    if not os.path.isdir(saveDir+"pdf/"): os.makedirs(saveDir+"pdf/")
    if not os.path.isdir(saveDir+"png/"): os.makedirs(saveDir+"png/")

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
    with open("Logs/" + modelType + "_rocLog.txt", "a") as rocLog: 
        spaces = " "*(30 - len(suffix))
        rocLog.write(suffix + ":" + spaces + "Avg: " + str(roc_auc_BES["macro"])[:8] + ", ")
        for sample in samples: 
            rocLog.write(sample + ": " + str(roc_auc_BES[sample])[:8] + ", ")
        rocLog.write("avg: " + str(roc_auc_BES["micro"])[:8] + "\n")


    # Plot ROC Curves
    for key, rocAUC in roc_auc_BES.items():

        # Assign these for readability:
        title = suffix + " " + labelDict[key][0] + " ROC Curve" 

        plt.figure(1)
        plt.plot(fprBES[key], tprBES[key],
                # label= 'BES ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                label= 'BES ROC Curve (area = ' + str(rocAUC)[:4] + ') ',
                color='orange', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

        path  = saveDir + "png/" + labelDict[key][1] + '_ROCplot.png'
        plt.savefig(path)
        path  = saveDir + "pdf/" + labelDict[key][1] + '_ROCplot.pdf'
        plt.savefig(path)
        plt.clf()
        plt.close()

#==================================================================================
# Plot pT Dependence //////////////////////////////////////////////////////////////
#==================================================================================
# This function needs to be updated later to flow with the final architecture. 
# As of now it would be too messy to try to implement it, but it should be added
#   to the normal pipeline in plotAll. 
# Once the final arch/vars are decided, can maybe add option for which var
#   to plot the dependence for. Could be completely general.
# def plotpTCM(BESpredict, truthLabels, pTArray, plotDir, suffix):
def plotpTCM(BESpredict, truthLabels, plotDir, args, year, testSet="flattened"):

    print("Plotting mistag rates and efficiency")
    sampleTypes = ["Ht","Wb","Zt","QCD"]

    # var:[binsize, xmin, xmax, xlabel, fileSuffix]
    plotDict = {"tot_HT":[50,1350,4500,"HT","HT"] }

    saveDir = os.path.join(plotDir,"tagging/") 
    if not os.path.isdir(saveDir+"pdf/"): os.makedirs(saveDir+"pdf/")
    if not os.path.isdir(saveDir+"png/"): os.makedirs(saveDir+"png/")
    # if not os.path.isdir(saveDir): os.makedirs(saveDir)

    # h5Path = "Sample_" + year + "_BESTinputs_test_" + args.suffix + ".h5" # This just makes the next few lines a bit more readable
    h5Path = "Sample_" + year + "_BESTinputs_test_" + testSet + ".h5" # This just makes the next few lines a bit more readable
    # _, varDict = loadMask(args.maskPath)
    allBesVarsList = "../formatConverter/h5samples/BESvarList.txt"
    _, varDict = loadMask(allBesVarsList)
    
    # for plotVar in plotDict.keys():
    for index, var in varDict.items():
        if not var in plotDict.keys(): continue

        binsize, xmin, xmax, xlabel, suff = plotDict[var]

        print("Loading " + var + " data")

        ptmassArray = np.concatenate([np.array(
            h5py.File(args.h5Dir + mySample + h5Path, "r")["BES_vars"])[...,int(index)] 
            for mySample in sampleTypes])
        
        # pT, mass, softdropmass, make dict, iterate over it, load arrays one at a time
        # need to grab from flattened/flattop, preScale

        cm = {}
        # bins_list = [i*binsize for i in range(xmin,xmax)]
        all_bins = [i*binsize for i in range(xmin,xmax)]
        suffix = args.suffix + "_" + year + "_" + suff

        # binsize = 25
        # bins_list = [i*binsize for i in range(20,64)]
        # # binsize = 5
        # # bins_list = [i*binsize for i in range(0,60)]
        # # suffix = suffix + "_mass"
        # # bins_list = [i*binsize for i in range(0,45)]
        # # suffix = suffix + "_SoftDrop"

        bins_list = []
        for pTmassbin in all_bins:       
            # print(pTmassbin)
            # Select events within certain pT range, create CM, save to dictionary
            ptmassIndex =  np.where(np.logical_and(ptmassArray >= pTmassbin, ptmassArray < (pTmassbin + binsize)))
            # print(ptmassIndex)
            # print(len(ptmassIndex))
            # print(np.array(ptmassIndex).shape)
            if len(ptmassIndex[0]) < 1: continue            
            bins_list.append(pTmassbin)
            cmTemp = metrics.confusion_matrix(truthLabels[ptmassIndex], np.argmax(BESpredict[ptmassIndex], axis=1), labels=[0,1,2,3,4,5] )
            # Normalize
            cm[pTmassbin] = cmTemp.astype('float') / cmTemp.sum(axis=1)[:, np.newaxis]
            
        targetNames = ['Ht',"Wb","Zt", 'QCD']
        for i, target in enumerate(targetNames):
            myPtArrays = [cm[pTmassbin][:,i] for pTmassbin in bins_list]
            # print(len(bins_list), len(myPtArrays))
            # print(bins_list, myPtArrays)

            # --- Create histogram, legend and title ---
            plt.figure()
            plt.plot(bins_list, myPtArrays)
            plt.legend(targetNames, title = "True Particle")
            plt.title("Percentage of X Classified as " + target + " Jets by " + xlabel + " for " + year)
            plt.xlabel("Jet " + xlabel + " (GeV)")
            plt.ylabel("Percentage of X Jets")
            plt.show()
            plt.savefig(os.path.join(saveDir, "png", suffix + '_Xas_' + target + '.png'))
            plt.savefig(os.path.join(saveDir, "pdf", suffix + '_Xas_' + target + '.pdf'))
            plt.clf()
            plt.close()

            myPtArrays = [cm[pTmassbin][i,:] for pTmassbin in bins_list]
            plt.figure()
            plt.plot(bins_list, myPtArrays)
            plt.legend(targetNames, title = "Classified As")
            plt.title("Percentage of " + target + " Jets Classified as X by " + xlabel + " for " + year)
            plt.xlabel("Jet " + xlabel + " (GeV)")
            plt.ylabel("Percentage of " + target + " Jets")
            plt.show()
            plt.savefig(os.path.join(saveDir, "png", suffix + '_' + target + '_asX.png'))
            plt.savefig(os.path.join(saveDir, "pdf", suffix + '_' + target + '_asX.pdf'))
            plt.clf()
            plt.close()

        del cm
        del myPtArrays
    print("Finished, check out your new plots at:")
    print(saveDir)

# def plotAll(args, strings, year, testSet):
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

    samples = ['Ht','Wb','Zt', 'QCD']
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

