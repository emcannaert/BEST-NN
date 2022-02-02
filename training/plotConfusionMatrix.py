#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plotConfusionMatrix.py //////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST with flattened inputs //////////////////////////////////
#==================================================================================

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

# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.models import load_model


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
sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
BatchSize = 1200
#models = ["BES","Images","Ensemble","Both"]
print("Begin CM")

def makeCM(model_BEST, h5Dir, plotDir, year, suffix, maskPath, testMaxEvents, modelType):
    import tools.functions as functs
    print("Begin CM")
    cm = {}
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
                           
    print("Plot confusion matrix")
    plt.figure()
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
        classifyMessage = "\t" + target + " Category: Tagger predicted\t" + str( totalPredicted ) + " " + str(np.sum(totalPredicted)) + " out of " + str(totalTested[i]) + " (" + str(100 * (float(totalPredicted[i])/float(totalTested[i]))  )[:6] + "%) truth events.\n"        
        print(classifyMessage)
        classifylog.write(classifyMessage)
        i += 1
    classifylog.write("-----------------------------------\n")
    classifylog.close
    print("Finished")


if __name__ == "__main__":
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
    args = parser.parse_args()

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

    # testMaxEvents = None
    testMaxEvents = 50000
    if os.path.isfile(modelFile):
        makeCM(load_model(modelFile), args.h5Dir, plotDir, args.year, mySuffix, maskSave, testMaxEvents, modelType)
