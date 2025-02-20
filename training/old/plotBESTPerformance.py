#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plotBESTPerformance.py //////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Samantha Abbott //////////////////////////////////////////////////////
# This program plots the Performance for a given BEST Model ///////////////////////
#==================================================================================

# user module 
import training.tools.functions_test as tools

# modules
import numpy as np
import tensorflow as tf
from sklearn import metrics

# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import load_model

sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
samples = ['W', 'Z', 'Higgs', 'Top', 'Bottom', 'QCD']

def plotAll(modelFile, historyFile, dataDict, plotDir, suffix, modelType, mask):
    print("Plotting BEST Performance")

    # Accuracy and Loss plots
    tools.plotAccLoss(historyFile, suffix, plotDir)
    
    eventData = dataDict["testEvents"][:,mask]
    # eventData = dataDict["testEvents"][()]
    truthData = dataDict["testTruth"]

    # Load model
    model_BEST = load_model(modelFile)
    print("Using BEST to predict...")
    BESpredict  = model_BEST.predict(eventData)
    del model_BEST

    # Plot ROC Curve
    tools.plotROC(BESpredict, truthData, plotDir, samples, modelType, suffix)

    # Collapse truth labels into 1D (length N_events) array containing truth index (0-5) for each event 
    truthData = np.argmax(truthData, axis=1) 

    # Plot Classification Probabilities
    tools.plotProbabilities(plotDir, BESpredict, truthData, samples)
    
    print("Making CM")
    print("My predictions shape:",      BESpredict.shape)
    print("Corresponding truth shape:", truthData.shape)
    
    cm = metrics.confusion_matrix(truthData, np.argmax(BESpredict, axis=1) )
                           
    # Plot Confusion Matrix, both normalized and not normalized
    tools.plot_confusion_matrix(cm, samples, plotDir, suffix)
    tools.plot_confusion_matrix(cm, samples, plotDir, suffix, normalize=True)

    # Record classification rates 
    tools.recordClassification(cm, truthData, samples, modelType, suffix)

    # Plot Efficiency
    # tools.plotpTCM(scaledTestEvents, truthData, testDataDict["testEvents"][:,548], plotDir, suffix)
    # tools.plotpTCM(BESpredict, truthData, plotDir, suffix)

    del BESpredict
    del truthData
    del dataDict
    del cm

    print("Finished Plotting BEST Performance. Check plots out at:")
    print(plotDir)

"""
if __name__ == "__main__":
    # need to update this part of the script in case one needs to plot a model
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="models/")
    parser.add_argument('-m','--maskPath',
                        dest='maskPath',
                        default="../formatConverter/masks/BESTMask.txt")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="")
    parser.add_argument('-sc','--scale',
                        dest='scale',
                        default="standardized")
    parser.add_argument('-y','--year',
                        dest='year',
                        default="2017")
    args = parser.parse_args()

    # modelType = "oldBEST"
    modelType = "nnBEST"

    # Load Mask
    mask = tools.loadMask(args.maskPath)

    # Load h5 Data, set up truth arrays
    dataDict = tools.loadH5Data(args.h5Dir, mask, sampleTypes, ["test"]) 

    # Generate appropriate helper strings
    modelDir, plotDir, suffix = tools.dirStrings(modelType, args, replaceModel=False)
    modelFile = modelDir + "BEST_model_" + suffix + ".h5"

    if os.path.isfile(modelFile):
        plotAll(load_model(modelFile), dataDict, plotDir, suffix, modelType)
    else: 
        print("Error, file does not exist: " + modelFile)
        quit()
"""
