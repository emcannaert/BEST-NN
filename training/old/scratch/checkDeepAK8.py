#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# checkDeepAK8.py //////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Sam Abbott ///////////////////////////////////////////////////////////
# This program plots the Confusion Matrix  //////////////////
#==================================================================================

################################## NOTES TO SELF ##################################


# modules
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import random
import numpy.random
import tensorflow as tf
from sklearn import metrics
from sklearn.externals.joblib import load
from sklearn import preprocessing

# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import load_model


# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
k.tensorflow_backend.set_session(tf.Session(config=config))

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
#print(sess.run(h))

# set options 
import tools.abbottFunctions as tools

sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
targetNames = ['W', 'Z', 'H', 'Top', 'b', 'QCD']

# def plotAll(model_BEST, h5Dir, plotDir, suffix, mask, modelType, scale):
# def plotAll(model_BEST, scaledDataDict, plotDir, suffix, modelType, scale):
def plotDeepAK8(dataDict, plotDir, modelType):
    import tools.abbottFunctions as tools

    suff   = "deepAK8"
    suffMD = "deepAK8MD"

    fixAK8MD = [4, 5, 1, 3, 0, 2]
    fixAK8   = [10, 11, 7, 9, 6, 8]
    deepAK8_Predict   = dataDict["testEvents"][:,fixAK8]
    deepAK8MD_Predict = dataDict["testEvents"][:,fixAK8MD]
    truthLabels       = dataDict["testTruth"]


    # Plot ROC Curve
    tools.plotROC(deepAK8_Predict,   truthLabels, plotDir, modelType, suff)
    tools.plotROC(deepAK8MD_Predict, truthLabels, plotDir, modelType, suffMD)

    # Collapse truth labels into 1D (length N_events) array containing truth index (0-5) for each event 
    truthLabels = np.argmax(truthLabels, axis=1) 

    # Plot Classification Probabilities
    tools.plotProbabilities(plotDir, deepAK8_Predict,   truthLabels, targetNames)
    tools.plotProbabilities(plotDir, deepAK8MD_Predict, truthLabels, targetNames)
    
    print("Making CM")
    print("deepAK8 predictions shape:",   deepAK8_Predict.shape   )
    print("deepAK8MD predictions shape:", deepAK8MD_Predict.shape )
    print("deepAK8 truth shape:",         truthLabels.shape       )
    
    cm   = metrics.confusion_matrix(truthLabels, np.argmax(deepAK8_Predict,   axis=1) )
    cmMD = metrics.confusion_matrix(truthLabels, np.argmax(deepAK8MD_Predict, axis=1) )
                           
    # Plot Confusion Matrix, both normalized and not normalized
    tools.plot_confusion_matrix(cm,   targetNames, plotDir, suff)
    tools.plot_confusion_matrix(cm,   targetNames, plotDir, suff,   normalize=True)
    tools.plot_confusion_matrix(cmMD, targetNames, plotDir, suffMD)
    tools.plot_confusion_matrix(cmMD, targetNames, plotDir, suffMD, normalize=True)

    # Record classification rates 
    tools.recordClassification(cm,   truthLabels, targetNames, modelType, suff)
    tools.recordClassification(cmMD, truthLabels, targetNames, modelType, suff)

    # Plot Efficiency
    # tools.plotpTCM(scaledTestEvents, truthTest, testDataDict["testEvents"][:,548], plotDir, suffix)
    # tools.plotpTCM(BESpredict, truthTest, plotDir, suffix)

    del deepAK8_Predict
    del deepAK8MD_Predict
    del truthLabels
    del cm
    del cmMD

    print("Finished")


if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/")
    args = parser.parse_args()

    modelType = "deepAK8"

    # MD = deepAK8Inds[0:6], Normal = deepAK8Inds[6:12]
    #                 B      H     QCD     T      W      Z
    deepAK8Inds = [ "530", "532", "533", "534", "535", "536",  # MD deepAK8
                    "538", "540", "541", "542", "543", "544" ] # Normal deepAK8
    
    mask = [True if str(i) in deepAK8Inds else False for i in range(596)]

    dataDict = tools.loadH5Data(args.h5Dir, mask, sampleTypes, ["test"]) 

    plotDir = "plots/" + modelType + "/"
    plotDeepAK8(dataDict, plotDir, modelType)



    #                               529:jetAK8_deepAK8MD_dnn_Largest
    # 530:jetAK8_deepAK8MD_rawB
    #                               531:jetAK8_deepAK8MD_rawC
    # 532:jetAK8_deepAK8MD_rawH
    # 533:jetAK8_deepAK8MD_rawL
    # 534:jetAK8_deepAK8MD_rawT
    # 535:jetAK8_deepAK8MD_rawW
    # 536:jetAK8_deepAK8MD_rawZ
    #                               537:jetAK8_deepAK8_dnn_Largest
    # 538:jetAK8_deepAK8_rawB
    #                               539:jetAK8_deepAK8_rawC
    # 540:jetAK8_deepAK8_rawH
    # 541:jetAK8_deepAK8_rawL
    # 542:jetAK8_deepAK8_rawT
    # 543:jetAK8_deepAK8_rawW
    # 544:jetAK8_deepAK8_rawZ
