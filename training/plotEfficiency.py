#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plotEfficiency.py ///////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Mark Samuel Abbott ///////////////////////////////////////////////////
# This program plots the tag rate and tag efficiency for a given BEST model ///////
#==================================================================================

################################## NOTES TO SELF ##################################
# Improve lables/titles.
# Tie this into plotConfusionMatrix.py? 
# Check for consistency, add comments.


# modules
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import copy
import random
import numpy.random

# get stuff from modules
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, SeparableConv2D, Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, MaxoutDense
from keras.layers import GRU, LSTM, ConvLSTM2D, Reshape
from keras.layers import concatenate
from keras.regularizers import l1,l2
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

# User definitons
# bins_list = [i*100 for i in range(0,40)]
binsize = 25
bins_list = [i*binsize for i in range(20,64)]

sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]

print("Begin pT Plotter")

def plotpTCM(model_BEST, h5Dir, plotDir, suffix, maskPath, testMaxEvents):
    # print("Before load")
    # print(globals().keys())
    print("Load model")
    cm = {}
    print(suffix)


    maskFile = open(maskPath, "r")
    maskIndex = []
    for line in maskFile:
        maskIndex.append(line.split(':')[0])
    maskFile.close()
    print(maskPath + " chosen; mask size " + str(len(maskIndex)))
    myMask = [True if str(i) in maskIndex else False for i in range(596)]


    myTestEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["BES_vars"])[:,myMask] for mySample in sampleTypes]
    print("My Test events shape:",[myTestEvents[i].shape for i in range(len(myTestEvents))])
    myTestTruth  = [np.zeros((len(myTestEvents[i]),len(sampleTypes))) for i in range(len(myTestEvents))] # shape: N,6 filled w/ 0s
    for i in range(len(sampleTypes)):
        myTestTruth[i][:,i] = 1.

    print("My test truth shape:",[myTestTruth[i].shape for i in range(len(myTestTruth))]) 
    print("Labels are:",[[i,mySample] for i,mySample in enumerate(sampleTypes)])
    globals()["jetBESvarsTest"]  = np.concatenate(myTestEvents)
    globals()["truthLabelsTest"] = np.concatenate(myTestTruth)


    print("Shuffle Test")
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["truthLabelsTest"])
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["jetBESvarsTest"])

    # testmask = [True if i == 548 else False for i in range(596)]
    # print("Plotting pT", suffix)
    # myPtArrays = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["BES_vars"])[:,548] for mySample in sampleTypes]
    myPtArrays = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened.h5","r")["BES_vars"])[:,548] for mySample in sampleTypes]
    # myPtArrays = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["BES_vars"])[:,testmask] for mySample in sampleTypes]
    globals()["jetptTest"]  = np.concatenate(myPtArrays)
    
    print("Shuffle pT")
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["jetptTest"])
    
    print("Max events to test on: " + str(testMaxEvents))
    print("(None means no limit, test all events)")
    allptArray = globals()["jetptTest"][:testMaxEvents]   

    print("Plotting pT")
    for pTbin in bins_list:       
        # print(str(pTbin) + " <= pT < " + str(pTbin + binsize))
        # Select events within certain pT range, create CM, save to dictionary
        # myPtArray = allptArray[(allptArray >= pTbin)*(allptArray < (pTbin + binsize))]
        ptIndex = np.where(np.logical_and(allptArray >= pTbin, allptArray < (pTbin + binsize)))
        # ptMask = numpy.zeros(testMaxEvents, dtype=bool)
        # for i in ptIndex: ptMask[i] = True

        # cm[pTbin] = metrics.confusion_matrix(np.argmax(model_BEST.predict([globals()["jetBESvarsTest"][ptMask] ]), axis=1), np.argmax(globals()["truthLabelsTest"][ptMask], axis=1) )
        # cmTemp = metrics.confusion_matrix(np.argmax(model_BEST.predict([globals()["jetBESvarsTest"][:testMaxEvents][ptIndex] ]), axis=1), np.argmax(globals()["truthLabelsTest"][:testMaxEvents][ptIndex], axis=1) )
        cmTemp = metrics.confusion_matrix(np.argmax(globals()["truthLabelsTest"][:testMaxEvents][ptIndex], axis=1), np.argmax(model_BEST.predict([globals()["jetBESvarsTest"][:testMaxEvents][ptIndex] ]), axis=1) )
        # Normalize
        # cmTemp = cmTemp.T
        cm[pTbin] = cmTemp.astype('float') / cmTemp.sum(axis=1)[:, np.newaxis]

    targetNames = ['W', 'Z', 'Higgs', 'Top', 'b', 'QCD']
    i = 0
    for target in targetNames:
        myPtArrays = [cm[pTbin][:,i] for pTbin in bins_list]

        # --- Create histogram, legend and title ---
        plt.figure()
        plt.plot(bins_list, myPtArrays)
        plt.legend(targetNames)
        plt.title("Tagging Rate for " + target )
        plt.xlabel("Jet pT (GeV)")
        plt.ylabel("Tagging Rate")
        plt.show()
        plt.savefig(plotDir + suffix + '_tagRate_' + target + '.png')
        plt.clf()

        myPtArrays = [cm[pTbin][i,:] for pTbin in bins_list]
        plt.figure()
        plt.plot(bins_list, myPtArrays)
        plt.legend(targetNames)
        plt.title("Tagging Efficiency for " + target )
        plt.xlabel("Jet pT (GeV)")
        plt.ylabel("Tagging Efficiency")
        plt.show()
        plt.savefig(plotDir + suffix + '_efficiency_' + target + '.png')
        plt.clf()

        i += 1

    print("Finished, check out your new plots at:")
    print(plotDir)


if __name__ == "__main__":
    #from johanTraining import loadData
    #loadData(args.h5Dir, args.year, ["Test"])
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="~/nobackup/models/")
    parser.add_argument('-m','--maskPath',
                        dest='maskPath',
                        default="/uscms/home/msabbott/nobackup/abbott/CMSSW_10_6_27/src/BEST/formatConverter/masks/oldBESTMask.txt")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="")
    parser.add_argument('-y','--year',
                        dest='year',
                        default="2017")
    #parser.add_argument('-m', '--models',
    #                     dest='models',
    #                     help='<Required> Which (comma separated) models to process. Examples: 1) all, 2) BES,Images,Combined,Ensemble',
    #                     required=True)
    #if not args.samples == "all": listOfSamples = args.samples.split(',')
    args = parser.parse_args()


    # mySuffix = args.suffix+args.year
    # mySuffix = args.suffix

    # modelType = "oldBEST"
    # modelType = "BESonly"
    modelType = "tweakedOldBEST"
    # modelType = ""
   
    maskName  = args.maskPath[1 + args.maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask
    mySuffix = args.suffix + args.year + "_" + maskName[:-4] + "_" + modelType
    plotDir  = "plots/" + modelType + "/" + mySuffix + "/PtDistributions/"
    modelDir = args.outDir + "/" + modelType + "/" + mySuffix
    modelFile = modelDir + "/BEST_model_" + mySuffix + ".h5"
    maskSave  = modelDir + "/" + maskName
    print(modelFile)

    if not os.path.isdir(plotDir): os.makedirs(plotDir)

    # for myModel in models:
    # testMaxEvents = None
    testMaxEvents = 50000
    if os.path.isfile(modelFile):
        plotpTCM(load_model(modelFile), args.h5Dir, plotDir, mySuffix, maskSave, testMaxEvents)
