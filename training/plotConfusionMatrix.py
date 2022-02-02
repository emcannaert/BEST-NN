#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plotConfusionMatrix.py //////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST with flattened inputs //////////////////////////////////
#==================================================================================

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

# user modules

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
#print(sess.run(h))

# set options 
savePDF = True
savePNG = True

setTypes = ["Test"]
sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
frameTypes = ["W","Z","Higgs","Top","Bottom"]

BatchSize = 1200

models = ["BES","Images","Ensemble","Both"]

print("Begin CM")

def loadData(h5Dir, year, setTypes, doBES, doImages):
    #==================================================================================
    # Initialize what will be np arrays ////////////////////////////////////////////////////////////
    #==================================================================================
    # This will create a series of global variables like jetTopFrameTrain and jetHiggsFrameValidation and jetBESvarsTrain, (4frames+1BesVars)*2sets=10globVars
    for mySet in setTypes:
        if doImages:
            for myFrame in frameTypes:
                globals()["jet"+myFrame+"Frame"+mySet] = []
        if doBES:
            globals()["jetBESvars"+mySet] = []

        globals()["truthLabels"+mySet] = []

    ## and this makes 12 global variables to store data

    print(globals().keys())

    #==================================================================================
    # Load Data from  h5 //////////////////////////////////////////////////////////////
    #==================================================================================

    # Loop over 2sets*6samples=12 files
    makeTruthLabelsOnce = True
    for mySet in setTypes:
        for index, mySample in enumerate(sampleTypes):
            print("Opening "+h5Dir+mySample+"Sample_"+year+"_BESTinputs_"+mySet.lower()+"_flattened_standardized.h5 file")
            myF = h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_"+mySet.lower()+"_flattened_standardized.h5","r")

            ## Make TruthLabels, only once (i.e. for key=BESvars)
            if globals()["truthLabels"+mySet] == []:
                print("Making new", "truthLabels"+mySet)
                globals()["truthLabels"+mySet] = np.full(len(myF['BES_vars'][()]), index)
            else:
                print("Concatenate", "truthLabels"+mySet)
                globals()["truthLabels"+mySet] = np.concatenate((globals()["truthLabels"+mySet], np.full(len(myF['BES_vars'][()]), index)))
            
            for myKey in myF.keys():
                varKey = "jet"
                if "image" in myKey.lower():
                    if not doImages:
                        continue
                    varKey = varKey+myKey.split("_")[0] # so HiggsFrame, TopFrame, etc
                else:
                    if not doBES:
                        continue
                    varKey = varKey+"BESvars"
               
            varKey = varKey+mySet
         
            ## Append data
            if globals()[varKey] == []:
               print("Making new", varKey)
               globals()[varKey] = myF[myKey][()]
            else:
               print("Concatenate", varKey)
               globals()[varKey] = np.concatenate((globals()[varKey], myF[myKey][()]))
            
            myF.close()
         
    print("Finished Accessing H5 data")
    ## Order of categories: 0-W, 1-Z, 2-H, 3-t, 4-b, 5-QCD (order of sampleTypes). Format properly.
    print("To_Categorical")
    for mySet in setTypes:
        globals()["truthLabels"+mySet] = to_categorical(globals()["truthLabels"+mySet], num_classes = 6)
        print("Made Truth Labels "+mySet, globals()["truthLabels"+mySet].shape)



    '''
    #==================================================================================
    # Initialize what will be np arrays ////////////////////////////////////////////////////////////
    #==================================================================================
    # This will create a series of global variables like jetTopFrameTrain and jetHiggsFrameValidation and jetBESvarsTrain, (4frames+1BesVars)*2sets=10globVars
    for mySet in setTypes:
        if doImages:
            for myFrame in frameTypes:
                globals()["jet"+myFrame+"Frame"+mySet] = []
        if doBES:
            globals()["jetBESvars"+mySet] = []
      
    #jetImagesTrain = [] #Should be a concatenation of XFrameImageTrain (ensure above sampleType order), each which appends {W,Z,H,Top,b,QCD}_XFrame_images_train
    #jetImagesValidation = [] #Should be a concatenation of XFrameImageValidation (ensure above sampleType order), each which appends {W,Z,H,Top,b,QCD}_XFrame_images_train

    truthLabelsTest = []

    ## and this makes 12 global variables to store data

    print(globals())


    #==================================================================================
    # Load Data from  h5 //////////////////////////////////////////////////////////////
    #==================================================================================

    # Loop over 2sets*6samples=12 files
    for mySet in setTypes:
        for index, mySample in enumerate(sampleTypes):
            print("Opening "+mySample+mySet+" file")
            myF = h5py.File("/uscms/home/bonillaj/nobackup/h5samples/"+mySample+"Sample_BESTinputs_"+mySet.lower()+"_flattened_standardized.h5","r")

            ## Make TruthLabels, only once (i.e. for key=BESvars)
            if globals()["truthLabels"+mySet] == []:
                print("Making new", "truthLabels"+mySet)
                globals()["truthLabels"+mySet] = np.full(len(myF["BES_vars"][()]), index)
            else:
                print("Concatenate", "truthLabels"+mySet)
                globals()["truthLabels"+mySet] = np.concatenate((globals()["truthLabels"+mySet], np.full(len(myF["BES_vars"][()]), index)))
      
            for myKey in myF.keys():
                varKey = "jet"
                if "image" in myKey.lower():
                    if not doImages:
                        continue
                    varKey = varKey+myKey.split("_")[0] # so HiggsFrame, TopFrame, etc
                else:
                    if not doBES:
                        continue
                    varKey = varKey+"BESvars"
               
                varKey = varKey+mySet
         
                ## Append data
                if globals()[varKey] == []:
                    print("Making new", varKey)
                    globals()[varKey] = myF[myKey][()]
                else:
                    print("Concatenate", varKey)
                    globals()[varKey] = np.concatenate((globals()[varKey], myF[myKey][()]))
            
            myF.close()
      
    print("Finished Accessing H5 data")

    ## Order of categories: 0-W, 1-Z, 2-H, 3-t, 4-b, 5-QCD (order of sampleTypes). Format properly.
    print("To_Categorical")
    truthLabelsTest = to_categorical(truthLabelsTest, num_classes = 6)
    print("Made Truth Labels Test", truthLabelsTest.shape)
    '''

def makeCM(model_BEST, h5Dir, plotDir, year, doBES, doImages, doEnsemble, suffix, maskPath, testMaxEvents, modelType):
    import tools.functions as functs
    #from johanTraining import loadData
    print("Before load")
    print(globals().keys())
    """
    if (doBES and not "jetBESvarsTest" in globals().keys()) or (doImages and not "jetWFrameTest" in globals().keys()):
        #loadData(h5Dir, year, ["Test"], doBES, doImages)
        myTestFile = h5py.File(outDir+"FinalSample_"+year+"_BESTinputs_test_flattened_standardized.h5","r")
        globals()["jetBESvarsTest"] = myTestFile["jetBESvarsTest"]
        globals()["jetWFrameTest"] = myTestFile["jetWFrameTest"]
        globals()["jetZFrameTest"] = myTestFile["jetZFrameTest"]
        globals()["jetHiggsFrameTest"] = myTestFile["jetHiggsFrameTest"]
        globals()["jetTopFrameTest"] = myTestFile["jetTopFrameTest"]
        globals()["jetBottomFrameTest"] = myTestFile["jetBottomFrameTest"]
    print("After load")
    print(globals().keys())
   
    if doBES:
        print("BESvars Test Shape", globals()["jetBESvarsTest"].shape)
    for myFrame in frameTypes:
        if not doImages or "Bottom" in myFrame:
            continue
        print(myFrame+" Images Train Shape", globals()["jet"+myFrame+"FrameTest"].shape)
   
    print("Shuffle Test")
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["truthLabelsTest"])
    if doBES:
        np.random.set_state(rng_state)
        np.random.shuffle(globals()["jetBESvarsTest"])
    if doImages:
        np.random.set_state(rng_state)
        np.random.shuffle(globals()["jetWFrameTest"])
        np.random.set_state(rng_state)
        np.random.shuffle(globals()["jetZFrameTest"])
        np.random.set_state(rng_state)
        np.random.shuffle(globals()["jetHiggsFrameTest"])
        np.random.set_state(rng_state)
        np.random.shuffle(globals()["jetTopFrameTest"])
    """
    print("Load model")
    cm = {}
    if doEnsemble:
        model_BES = load_model(h5Dir+"BEST_model"+suffix.split("_Ensemble")[0]+"_BES.h5")
        #model_BES = load_model("/uscms/home/bonillaj/nobackup/models/BEST_model_BES.h5")
        model_Images = load_model(h5Dir+"BEST_model"+suffix.split("_Ensemble")[0]+"_Images.h5")
        #model_Images = load_model("/uscms/home/bonillaj/nobackup/models/BEST_model_Images.h5")
        print("Making BES test predictions")
        predictTestBES = model_BES.predict([globals()["jetBESvarsTest"][:]])
        print("Making image test predictions")
        predictTestImages = model_Images.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:]])
        #model_BEST = load_model(modelBEST)
        print("Make confusion matrix")
        cm["BES"] = metrics.confusion_matrix(np.argmax(model_BES.predict([globals()["jetBESvarsTest"][:] ]), axis=1), np.argmax(globals()["truthLabelsTest"][:], axis=1) )
        cm["Images"] = metrics.confusion_matrix(np.argmax(model_Images.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:]]), axis=1), np.argmax(globals()["truthLabelsTest"][:], axis=1) )
        cm["Ensemble"] = metrics.confusion_matrix(np.argmax(model_BEST.predict([np.concatenate((predictTestBES[:], predictTestImages[:]), axis=1)]), axis=1), np.argmax(globals()["truthLabelsTest"][:], axis=1) )
    else:
        # #model_BEST = load_model(modelBEST)
            if doBES and not doImages:
                print(suffix)
                # if 'oldBEST' in suffix:
                #     maskFile = open("/uscms/home/msabbott/nobackup/abbott/CMSSW_10_6_27/src/BEST/formatConverter/masks/oldBESTMask.txt", "r")
                #     oldBESTMaskIndex = []
                #     for line in maskFile:
                #         oldBESTMaskIndex.append(line.split(':')[0])
                #     maskFile.close()
                #     print("Old BEST Mask size", len(oldBESTMaskIndex))
                #     myMask = [True if str(i) in oldBESTMaskIndex else False for i in range(596)]
                # else:
                #     myMask = [True for i in range(123)]
                #     # myMask = [True for i in range(596)]

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

                print("Making CM")
                print("Max events to test on: " + str(testMaxEvents))
                print("(None means no limit, test all events)")
                # cm["BES"] = metrics.confusion_matrix(np.argmax(model_BEST.predict([globals()["jetBESvarsTest"][:] ]), axis=1), np.argmax(globals()["truthLabelsTest"][:], axis=1) )
                cm["BES"] = metrics.confusion_matrix(np.argmax(globals()["truthLabelsTest"][:testMaxEvents], axis=1), np.argmax(model_BEST.predict([globals()["jetBESvarsTest"][:testMaxEvents] ]), axis=1) )
                # cm["BES"] = metrics.confusion_matrix(np.argmax(model_BEST.predict([globals()["jetBESvarsTest"][:testMaxEvents] ]), axis=1), np.argmax(globals()["truthLabelsTest"][:testMaxEvents], axis=1) )
               
                # else:
                #    myTestFile = h5py.File(h5Dir+"FinalSampleCompressedAllImages_2017_test_flattened_standardized_shuffled_v4.h5","r")
                #    print("v4")
                #    cm["BES"] = metrics.confusion_matrix(np.argmax(model_BEST.predict([myTestFile["jetBESvarstest"][:] ]), axis=1), np.argmax(myTestFile["truthLabelstest"][:], axis=1) )
            elif not doBES and doImages:
                """
                print("Test Images shape for single frame:",globals()["jetWFrameTest"].shape)
                concatStepTest = int(math.ceil(globals()["jetWFrameTest"].shape[0]/10.))
                print("ConcatStepTest:",concatStepTest)
                testImages = []#np.zeros((globals()["jetWFrameTrain"].shape[0],31,31,4), dtype=float)
                for i in range(0,10):
                    print("Test Step",i)
                    eventLowTest = i*concatStepTest
                    eventHighTest = min(globals()["jetWFrameTest"].shape[0],(i+1)*concatStepTest)
                    print("Events:",eventLowTest,eventHighTest)
                    if i == 0:
                        testImages = tf.keras.layers.Concatenate(axis=3)([globals()["jetWFrameTest"][eventLowTest:eventHighTest], globals()["jetZFrameTest"][eventLowTest:eventHighTest], globals()["jetHiggsFrameTest"][eventLowTest:eventHighTest], globals()["jetTopFrameTest"][eventLowTest:eventHighTest]])[()]
                    else:
                        newSet = tf.keras.layers.Concatenate(axis=3)([globals()["jetWFrameTest"][eventLowTest:eventHighTest], globals()["jetZFrameTest"][eventLowTest:eventHighTest], globals()["jetHiggsFrameTest"][eventLowTest:eventHighTest], globals()["jetTopFrameTest"][eventLowTest:eventHighTest]])[()]
                        print("Sizes:",testImages.shape,newSet.shape)
                        testImages = tf.keras.layers.Concatenate(axis=0)([testImages[:], newSet[:]])[()]
                    print("After concat shape test", testImages.shape)
            
                cm["Images"] = metrics.confusion_matrix(np.argmax(model_BEST.predict(testImages), axis=1), np.argmax(globals()["truthLabelsTest"][:], axis=1) )
                """
                """
                for myFrame in frameTypes:
            
                    myTestEvents = [] # format: for each of 5 frames new 6samplesxNEventsInSamplex31pixelsx31pixelsx1float
                    if not "truthLabelsTrain" in globals():
                        myTestTruth = [] # format: 6samplesx6categoriesx1float
                
                    for mySample in sampleTypes:
                        myTF = h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")
                        NTEvents = myTF[myFrame+"Frame_images"].shape[0]
                        myTShape=(NTEvents,31,31,1)
                        myTestEvents.append(np.array(myTF[myFrame+"Frame_images"][...,0]).resize(myTShape))
                        myTF.close()
                        if not "truthLabelsTest" in globals():
                            myTestTruth.append(np.zeros((NTEvents,len(sampleTypes)))) # shape: NEventsInSamplex6Categoriesx1float filled w/ 0s
                
                    globals()["jet"+myFrame+"FrameTest"] = np.concatenate(myTestEvents)
                    if not "truthLabelsTest" in globals():
                        for i in range(len(sampleTypes)):
                            myTestTruth[i][:,i] = 1.
                        globals()["truthLabelsTest"] = np.concatenate(myTestTruth)
                """

                myTestWEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["WFrame_images"][...,0]) for mySample in sampleTypes]
                myTestZEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["ZFrame_images"][...,0]) for mySample in sampleTypes]
                myTestHiggsEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["HiggsFrame_images"][...,0]) for mySample in sampleTypes]
                myTestTopEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["TopFrame_images"][...,0]) for mySample in sampleTypes]
                myTestBottomEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["BottomFrame_images"][...,0]) for mySample in sampleTypes]
                myTestTruth = [np.zeros((len(myTestWEvents[i]),len(sampleTypes))) for i in range(len(myTestWEvents))] # shape: N,6 filled w/ 0s
                for i in range(len(sampleTypes)):
                    myTestTruth[i][:,i] = 1.
         
                print("My test truth shape:",[myTestTruth[i].shape for i in range(len(myTestTruth))])
                print("Labels are:",[[i,mySample] for i,mySample in enumerate(sampleTypes)])
                globals()["jetWFrameTest"] = np.concatenate(myTestWEvents)
                globals()["jetZFrameTest"] = np.concatenate(myTestZEvents)
                globals()["jetHiggsFrameTest"] = np.concatenate(myTestHiggsEvents)
                globals()["jetTopFrameTest"] = np.concatenate(myTestTopEvents)
                globals()["jetBottomFrameTest"] = np.concatenate(myTestBottomEvents)
                globals()["truthLabelsTest"] = np.concatenate(myTestTruth)
         
                totalEventsTest = len(globals()["truthLabelsTest"])
         
                cm["Images"] = metrics.confusion_matrix(
                    np.argmax(
                        model_BEST.predict(
                            [globals()["jetWFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetZFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetHiggsFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetTopFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetBottomFrameTest"].reshape((totalEventsTest,31,31,1))
                            ]),
                        axis=1
                    ),
                    np.argmax(globals()["truthLabelsTest"].reshape((totalEventsTest,6)), axis=1)
                )
                #cm["Images"] = metrics.confusion_matrix(np.argmax(model_BEST.predict([myTestFile["jetWFrametest"][:], myTestFile["jetZFrametest"][:], myTestFile["jetHiggsFrametest"][:], myTestFile["jetTopFrametest"][:]]), axis=1), np.argmax(myTestFile["truthLabelstest"][:], axis=1) )
            elif doBES and doImages:

                myTestWEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["WFrame_images"][...,0]) for mySample in sampleTypes]
                myTestZEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["ZFrame_images"][...,0]) for mySample in sampleTypes]
                myTestHiggsEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["HiggsFrame_images"][...,0]) for mySample in sampleTypes]
                myTestTopEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["TopFrame_images"][...,0]) for mySample in sampleTypes]
                myTestBottomEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["BottomFrame_images"][...,0]) for mySample in sampleTypes]
                myTestBESEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")["BES_vars"]) for mySample in sampleTypes]
                myTestTruth = [np.zeros((len(myTestWEvents[i]),len(sampleTypes))) for i in range(len(myTestWEvents))] # shape: N,6 filled w/ 0s
                for i in range(len(sampleTypes)):
                    myTestTruth[i][:,i] = 1.
         
                print("My test truth shape:",[myTestTruth[i].shape for i in range(len(myTestTruth))])
                print("Labels are:",[[i,mySample] for i,mySample in enumerate(sampleTypes)])
                globals()["jetWFrameTest"] = np.concatenate(myTestWEvents)
                globals()["jetZFrameTest"] = np.concatenate(myTestZEvents)
                globals()["jetHiggsFrameTest"] = np.concatenate(myTestHiggsEvents)
                globals()["jetTopFrameTest"] = np.concatenate(myTestTopEvents)
                globals()["jetBottomFrameTest"] = np.concatenate(myTestBottomEvents)
                globals()["jetBESvarsTest"] = np.concatenate(myTestBESEvents)
                globals()["truthLabelsTest"] = np.concatenate(myTestTruth)
         
                totalEventsTest = len(globals()["truthLabelsTest"])
         
                cm["ImagesAndBES"] = metrics.confusion_matrix(
                    np.argmax(
                        model_BEST.predict(
                            [globals()["jetWFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetZFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetHiggsFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetTopFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetBottomFrameTest"].reshape((totalEventsTest,31,31,1)),
                            globals()["jetBESvarsTest"].reshape((totalEventsTest,142))
                            ]),
                        axis=1
                    ),
                    np.argmax(globals()["truthLabelsTest"].reshape((totalEventsTest,6)), axis=1)
                )
         
                #myTestFile = h5py.File(h5Dir+"FinalSampleCompressedAllImages_2017_test_flattened_standardized_shuffled_v4.h5","r")
                #cm["Combined"] = metrics.confusion_matrix(np.argmax(model_BEST.predict([myTestFile["jetWFrametest"][:], myTestFile["jetZFrametest"][:], myTestFile["jetHiggsFrametest"][:], myTestFile["jetTopFrametest"][:], myTestFile["jetBottomFrametest"][:], myTestFile["jetBESvarstest"][:] ]), axis=1), np.argmax(myTestFile["truthLabelstest"][:], axis=1) )
            print("Plot confusion matrix")
    plt.figure(
    )
    targetNames = ['W', 'Z', 'Higgs', 'Top', 'b', 'QCD']
    for myKey in cm.keys():
        print("myKey",myKey)

        # functs.plot_confusion_matrix(cm[myKey].T, targetNames)
        functs.plot_confusion_matrix(cm[myKey], targetNames)
        if savePDF == True:
            if not os.path.isdir(plotDir): os.makedirs(plotDir)
            print("Saving to", plotDir+'/ConfusionMatrix_'+myKey+suffix+'.pdf')
            plt.savefig(plotDir+'/ConfusionMatrix_'+myKey+suffix+'.pdf')
        plt.clf()

        # functs.plot_confusion_matrix(cm[myKey].T, targetNames, normalize=True)
        functs.plot_confusion_matrix(cm[myKey], targetNames, normalize=True)
        if savePDF == True:
            if not os.path.isdir(plotDir): os.makedirs(plotDir)
            print("Saving to", plotDir+'/ConfusionMatrix_'+myKey+suffix+'_normalized.pdf')
            plt.savefig(plotDir+'/ConfusionMatrix_'+myKey+suffix+'_normalized.pdf')
        plt.clf()
    plt.close()

    # Record classification rates
    classifylog = open("logs/classifylog_" + modelType, "a") 
    classifylog.write("-----------------------------------\n")
    classifylog.write("Running " + suffix + ":\n")
    totalTested = np.count_nonzero(globals()["truthLabelsTest"][:testMaxEvents] == 1, axis=0)
    i = 0
    targetNames = ['W', 'Z', 'H', 'Top', 'b', 'QCD']
    classifylog.write("\t\t\t\t\t\t\t\t\t\tW\t Z\t  H\t  Top\tb  QCD  Total\n")
    for target in targetNames:
        # totalPredicted = cm["BES"][i][i]
        # classifyMessage = "\t" + target + " Category: Tagger predicted\t" + str( totalPredicted ) + "/" + str(totalTested[i]) + " events " + "(" + str(100 * (totalPredicted/totalTested[i]) )[:6] + "%). Tagged " + np.sum()+ " total events as " + target + ".\n"
        # classifylog.write("\t" + target + " Category: Tagger predicted\t" + str( totalPredicted ) + "/" + str(totalTested[i]) + " events " + "(" + str(100 * (totalPredicted/totalTested[i]) )[:6] + "%).\n")
        totalPredicted = cm["BES"][:,i]
        # print(totalPredicted[i])
        # print(totalTested[i] )
        # print(100 * (float(totalPredicted[i])/float(totalTested[i])) )
        classifyMessage = "\t" + target + " Category: Tagger predicted\t" + str( totalPredicted ) + " " + str(np.sum(totalPredicted)) + " out of " + str(totalTested[i]) + " (" + str(100 * (float(totalPredicted[i])/float(totalTested[i]))  )[:6] + "%) truth events.\n"        
        print(classifyMessage)
        classifylog.write(classifyMessage)
        i += 1
    classifylog.write("-----------------------------------\n")
    classifylog.close
    print("Finished")


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
    parser.add_argument('-b','--doBES', dest='doBES', default=False, action='store_true')
    parser.add_argument('-i','--doImages', dest='doImages', default=False, action='store_true')
    parser.add_argument('-e','--doEnsemble', dest='doEnsemble', default=False, action='store_true')
    args = parser.parse_args()

    doBES = args.doBES
    doImages = args.doImages
    doEnsemble = args.doEnsemble
    if doEnsemble:
        doBES = True
        doImages = True
    # mySuffix = args.suffix+args.year
    # mySuffix = args.suffix

    # modelType = "oldBEST"
    # modelType = "BESonly"
    modelType = "tweakedOldBEST"
    # modelType = ""
   
    maskName  = args.maskPath[1 + args.maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask
    mySuffix = args.suffix + args.year + "_" + maskName[:-4] + "_" + modelType
    plotDir  = "plots/" + modelType + "/" + mySuffix
    modelDir = args.outDir + "/" + modelType + "/" + mySuffix
    modelFile = modelDir + "/BEST_model_" + mySuffix + ".h5"
    maskSave  = modelDir + "/" + maskName
    print(modelFile)

    # for myModel in models:
    # testMaxEvents = None
    testMaxEvents = 50000
    if os.path.isfile(modelFile):
        makeCM(load_model(modelFile), args.h5Dir, plotDir, args.year, doBES, doImages, doEnsemble, mySuffix, maskSave, testMaxEvents, modelType)
