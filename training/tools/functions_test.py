#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# functions.py ////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Samantha Abbott, Reyer Band, Johan S. Bonilla, Brendan Regnary ///////
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
    print(timeMessage+"\n")
    with open("Logs/timeLog.txt", "a") as f:
        f.write(timeMessage)

#==================================================================================
# Plot Confusion Matrix ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# cm is the comfusion matrix //////////////////////////////////////////////////////
# classes are the names of the classes that the classifier distributes among //////
#----------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes, plotDir, suffix, year,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print("Plot confusion matrix")
    saveDir = os.path.join(plotDir,"ConfusionMatrix/")
    if not os.path.isdir(saveDir): os.makedirs(saveDir)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix, normalized")
        title = "Normalized Confusion Matrix"
        # saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix + '_normalized.png')
        saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix + '_normalized')
    else:
        print('Confusion matrix, without normalization')
        title = "Confusion Matrix"
        # saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix + '.png')
        saveFile = os.path.join(saveDir, 'ConfusionMatrix_BES' + suffix)

    print(cm)
    if (suffix=="2015"):
        title = "2016_preAPV"+ " " + title
    elif (suffix=="2016"):
        title= "2016_postAPV"+ " " + title
    else:
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

    print("Saving to: " + saveFile + "_%s.png"%year)
    plt.savefig(saveFile + ".png")
    print("Saving to: " + saveFile + "_%s.pdf"%year)
    plt.savefig(saveFile + ".pdf")

    plt.clf()
    plt.close()

#==================================================================================
# Plot Performance ////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# loss is an array of loss and loss_val from the training /////////////////////////
# acc is an array of acc and acc_val from the training ////////////////////////////
#----------------------------------------------------------------------------------

def plotAccLoss(historyFile, suffix, plotDir, year): 
    
    
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
    plt.plot(loss, label='loss; Min loss: ' + str(np.min(loss))[:6] + ', Epoch: ' + str(np.argmin(loss)) )
    plt.plot(val_loss, label='val_loss; Min val_loss: ' + str(np.min(val_loss))[:6] + ', Epoch: ' + str(np.argmin(val_loss)) )
    if (suffix=="2015"):
        plt.title("2016_preAPV" + " loss and val_loss vs. epochs")
    elif (suffix=="2016"):
        plt.title("2016_postAPV" + " loss and val_loss vs. epochs")
    else:
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
    plt.plot(acc,     label='acc; Max acc: '  + str(np.max(acc))[:6] + ', Epoch: ' + str(np.argmax(acc)) )
    plt.plot(val_acc, label='val_acc; Max val_acc: ' + str(np.max(val_acc))[:6] + ', Epoch: ' + str(np.argmax(val_acc)) )
    if (suffix=="2015"):
        plt.title("2016_preAPV" + " acc and val_acc vs. epochs")
    elif (suffix=="2016"):
        plt.title("2016_postAPV" + " acc and val_acc vs. epochs")
    else:
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
        if (year=="2015"):
            title = "Probability of " + target + " Classification for " + "2016_preAPV"
        elif (year=="2016"):
            title= "Probability of " + target + " Classification for " + "2016_postAPV"
        else:
            title = "Probability of " + target + " Classification for " + year
        for j, mylabel in enumerate(targetNames):
            tempMask = [True if j == k else False for k in truthTest]
            plt.hist(eventPredictions[tempMask,0], label = mylabel, bins = 20, range = (0,1), histtype='step', log = True)
        plt.xlim(0.0,1.0)
        leg = plt.legend(ncol = 6, loc = 'upper center', bbox_to_anchor = (0.0,1.1,1.0,0.1), borderpad = 0.9, borderaxespad = 2.0 )
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(1.1)
        plt.xlabel( title )
        plt.gca().tick_params(axis = 'y', which = 'both', direction = 'in', left = True, right = True)
        plt.gca().tick_params(axis = 'x', direction = 'in', top = True, bottom = True)
        plt.show()
        plt.savefig(saveDir + "_".join(title.split(" ")) + ".png")
        plt.savefig(saveDir + "_".join(title) + ".pdf")
        plt.clf()
    plt.close()

#==================================================================================
# Load Mask ///////////////////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Loads Mask text file and returns Boolean masking array //////////////////////////
#----------------------------------------------------------------------------------

# def loadMask(maskPath, max = 551):
# def loadMask(maskPath, max = 270):
def loadMask(maskPath, max = 113):
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

def loadH5Data(args, mask, sampleTypes, setTypes, maxEvents, mass_type, year='', testSet="flattened"):
    h5Dir = args.h5Dir
    scale = args.scale
    if year == '': year = args.year

    if not scale == "": scale = "_" + scale
    dataDict = {}
    for mySet in setTypes:

        numEvents = maxEvents[mySet]
        print("Loading h5 files for " + mySet)

        mass_str= mass_type + "_"
        h5Path = "_Sample_"+ mass_str +year+'_BESTinputs_' + mySet + "_" + args.suffix + scale + ".h5" # This just makes the next few lines a bit more readable

        # h5Path = "_Sample_"+ mass_str +year+'_BESTinputs_' + mySet + "_1" + "_" + args.suffix + scale + ".h5" # This just makes the next few lines a bit more readable
        # if mySet == "test": h5Path = "_Sample_" + mass_str +year+"_BESTinputs_" + mySet + "_" + testSet + ".h5"
        print(h5Path)
        
        
        # Check if loading all variables. Quicker to NOT use mask in this case
        
        if np.all(mask):
            eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[:numEvents,:]     for mySample in sampleTypes]
        else:

            #print("numEvents is ", numEvents)
            #print("mask is ", mask)
            #print([sample for sample in sampleTypes])
            #print("length of that is %i"%len([sample for sample in sampleTypes]) )
            #print( np.array(h5py.File(h5Dir + "QCD" + h5Path, "r")["BES_vars"])[0,:]    )
            eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[:numEvents,mask] for mySample in sampleTypes]

        print("My " + mySet + " events shape:", [eventArrays[i].shape for i in range(len(eventArrays))])

        # Create Truth arrays; shape: N_events x 6
        truthArrays = [np.zeros( (len(eventArrays[i]), 1) ) for i in range(len(eventArrays))] 

        # Arrays are filled with zeros. Now set 1's to record Truth particle info
        for i in range(len(sampleTypes)):
            print("sampletype", sampleTypes[i])
            if sampleTypes[i] == "bg":
                truthArrays[i][:, 0] = 0  
            else:
                truthArrays[i][:, 0] = 1 
        print(truthArrays)        

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

    # scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    # scalePath = scaleDir + "ScalerParameters_" + scale + ".joblib"

    scalePath = "/uscms/home/cannaert/nobackup/newBEST/CMSSW_10_6_27/src/BEST/training/ScalerParameters/newBEST_Basic.joblib"
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
    truthTest = truthTest.astype(int)
    totalTested = np.bincount(truthTest)
    classifylog.write("\t\t\t\t\t\t\t\t\t\tW\t Z\t  H\t  Top\tb  QCD  Total\n")
    for i, target in enumerate(targetNames):
        if cm.shape[1] > 1:
            totalPredicted = cm[:, i]
        else:
            totalPredicted = cm[:, 0]
        print("cm shape", cm.shape)
        classifyMessage = "\t" + target + " Category: Tagger predicted\t" + str( totalPredicted ) + " " + str(np.sum(totalPredicted)) + " out of " + str(totalTested[i]) + " (" + str(100 * (float(totalPredicted[i])/float(totalTested[i]))  )[:6] + "%) truth events.\n"        
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

def dirStrings(args, nLayers, nNodes, nCats, mass_type, year=''):
    # Create useful strings
    maskName = args.maskPath[1 + args.maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask
    # mySuffix = args.suffix + args.scale + maskName[11:-4] 
    if year == '': year = args.year
    # mySuffix = args.suffix + "_" + year
    mySuffix = year

    nodes_str = nNodes + "nodes_" # number of nodes
    layers_str = nLayers + "layers_" # number of strings 
    cats_str  = nCats + "cats_" #number of categories
    plotDir  = "plots/"    + args.modelType+ "_" + nLayers + "Layers_" + nNodes + "Nodes_" + nCats + "Categories/" + mySuffix + "/" + mass_type + "/"

    #             models/   nnBEST             _80             Layers_      3      Nodes_  all_mass   Categories    /2016/                12/
    modelDir = args.outDir + args.modelType + "_" + nLayers + "Layers_" + nNodes + "Nodes_" + nCats + "Categories/" + mySuffix + "/" + mass_type + "/"
    
    modelFile = modelDir + "BEST_model_" + mass_type + "_" + cats_str+ layers_str+ nodes_str+  mySuffix + ".h5"
    # modelFile = modelDir + "BEST_model_" + mySuffix + ".pb"
    maskSave  = modelDir + maskName    
    historyFile  = modelDir + "history_" + mass_type + "_" + mySuffix + ".joblib" 

    print("Looking for history file:", historyFile)
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

def plotROC(BESpredict, truthLabels, plotDir, samples, modelType, suffix, year):

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
        print("i is %i, sample is %s"%(i,sample))
        fprBES[sample], tprBES[sample], _ = roc_curve(truthLabels[:, 0], BESpredict[:, 0])
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
        if (suffix=="2015"):
            title = "2016_preAPV"+ " " + labelDict[key][0] + " ROC Curve" 
        elif (suffix=="2016"):
            title= "2016_postAPV"+ " " + labelDict[key][0] + " ROC Curve" 
        else:
            title = suffix + " " + labelDict[key][0] + " ROC Curve"  
        plt.figure(1)
        plt.plot(fprBES[key], tprBES[key],
                # label= 'BES ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                label= 'BES ROC Curve (area = ' + str(rocAUC)[:6] + ') ',
                color='orange', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")

        path  = saveDir + "png/" + labelDict[key][1] + '_ROCplot_%s.png'%year
        plt.savefig(path)
        path  = saveDir + "pdf/" + labelDict[key][1] + '_ROCplot_%s.png'%year
        # plt.savefig(path)
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
def plotpTCM(BESpredict, truthLabels, plotDir, args, mass_type, year, sampleTypes, testSet="flattened"):

    print("Plotting mistag rates and efficiency")
    #sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]

    # var:[binsize, xmin, xmax, xlabel, fileSuffix]
    # var:[binsize, xmin / binsize, xmax / binsize, xlabel, fileSuffix]
    plotDict = {"tot_HT":[100,16,98,"HT","event_HT"],
                "SJ_mass":[100,0,40,"superjet mass","SJ_mass"] }

    saveDir = os.path.join(plotDir,"tagging/") 
    if not os.path.isdir(saveDir+"pdf/"): os.makedirs(saveDir+"pdf/")
    if not os.path.isdir(saveDir+"png/"): os.makedirs(saveDir+"png/")
    # if not os.path.isdir(saveDir): os.makedirs(saveDir)

    mass_str= mass_type + "_"

    # h5Path = "Sample_" + year + "_BESTinputs_test_" + args.suffix + ".h5" # This just makes the next few lines a bit more readable
    h5Path = "_Sample_"+ mass_str +year+ "_BESTinputs_test_" + testSet + ".h5" # This just makes the next few lines a bit more readable
    # _, varDict = loadMask(args.maskPath)
    # allBesVarsList = "../formatConverter/h5samples/BESvarList_noHT_noEventNum.txt"
    allBesVarsList = "../formatConverter/h5samples/BESvarList_full.txt"
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
        all_bins = [i*binsize for i in range(xmin,xmax+1)]
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
            # print(len(ptmassIndex))
            # print(np.array(ptmassIndex).shape)
            if len(ptmassIndex[0]) < 100: continue    
         
            bins_list.append(pTmassbin)
            predicted_labels = (BESpredict >= 0.5).astype(int)
            cmTemp = metrics.confusion_matrix(truthLabels[ptmassIndex],predicted_labels[ptmassIndex],labels=[0,1] )
        
            # Normalize
            cm[pTmassbin] = cmTemp.astype('float') / cmTemp.sum(axis=1)[:, np.newaxis]
        
        targetNames = sampleTypes
        for i, target in enumerate(targetNames):
            myPtArrays = [cm[pTmassbin][:,i] for pTmassbin in bins_list]
            # print(len(bins_list), len(myPtArrays))
            # print(bins_list, myPtArrays)

            # --- Create histogram, legend and title ---
            plt.figure()
            plt.plot(bins_list, myPtArrays)
            plt.legend(targetNames, title = "True Particle")
            if (year=="2015"):
                plt.title("Percentage of X Classified as " + target + " Jets by " + xlabel + " for " + "2016_preAPV")
            elif (year=="2016"):
                plt.title("Percentage of X Classified as " + target + " Jets by " + xlabel + " for " + "2016_postAPV")
            else:
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
            if (year=="2015"):
                plt.title("Percentage of " + target + " Jets Classified as X by " + xlabel + " for " + "2016_preAPV")
            elif (year=="2016"):
                plt.title("Percentage of " + target + " Jets Classified as X by " + xlabel + " for " + "2016_postAPV")
            else:
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
def plotAll(args, strings, truthData, modelType, BESpredict, mass_type, year, samples):
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
    plotAccLoss(historyFile, suffix, plotDir, year)
    
    #samples = ["WB","HT","ZT","Top","QCD"]

    # sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]

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
    plotROC(BESpredict, truthData, plotDir, samples, args.modelType, suffix, year)

    # Collapse truth labels into 1D (length N_events) array containing truth index (0-5) for each event 
    truthData = truthData.ravel()

    # Plot Classification Probabilities
    # plotProbabilities(plotDir, BESpredict, truthData, samples)
    plotProbabilities(plotDir, BESpredict, truthData, samples, year)
    
    print("Making CM")
    print("My predictions shape:",      BESpredict.shape)
    print(BESpredict)
    print("Corresponding truth shape:", truthData.shape)
    print(truthData)

    BESpredict_flat = BESpredict.flatten()
    BESpredict_flat_signal = BESpredict_flat[truthData == 1]
    BESpredict_flat_bg = BESpredict_flat[truthData == 0]

    # Plot the histogram
    plt.hist(BESpredict_flat_signal, bins=20, edgecolor='black')
    plt.title('Histogram of signal BESpredict')
    plt.xlabel('Prediction Values')
    plt.ylabel('Events')
    plt.savefig('BESpredict_histogram_sig.png')
    plt.show()
    plt.clf()

    plt.hist(BESpredict_flat_bg, bins=20, edgecolor='black')
    plt.title('Histogram of background BESpredict')
    plt.xlabel('Prediction Values')
    plt.ylabel('Events')
    plt.savefig('BESpredict_histogram_bg.png')
    plt.show()
    

    predicted_labels = (BESpredict >= 0.5).astype(int)
    cm = metrics.confusion_matrix(truthData,predicted_labels )
    print("Confusion Matrix:")
    print(cm)
                           
    # Plot Confusion Matrix, both normalized and not normalized
    plot_confusion_matrix(cm, samples, plotDir, suffix, year)
    plot_confusion_matrix(cm, samples, plotDir, suffix, year, normalize=True)
    # plot_confusion_matrix(cm, samples, plotDir, suffix, normalize=True, compare=True, args=args, year=year, testSet=testSet)

    # Record classification rates 
    # recordClassification(cm, truthData, samples, modelType, suffix)
    recordClassification(cm, truthData, samples, args.modelType, suffix)

    # Plot Efficiency
    plotpTCM(predicted_labels, truthData, plotDir, args, mass_type, year, samples)
    # plotpTCM(BESpredict, truthData, plotDir, args, year, testSet)
    # plotpTCM(scaledTestEvents, truthData, testDataDict["testEvents"][:,548], plotDir, suffix)


    print("Finished Plotting BEST Performance. Check plots out at:")
    print(plotDir)