import time
# startTime = time.time() # Tracks how long script takes




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
from sklearn import metrics

# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
k.tensorflow_backend.set_session(tf.Session(config=config))

import h5py
import numpy as np
import numpy.random
import tools.functions as tools
sampleTypes = ["Zt","Ht","Wb","QCD1500to2000","QCD2000toInf","TTBar"]
h5Dir="/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/formatConverter/h5samplesSplit/"
maskPath = '../formatConverter/masks/BESvarList.txt'
setTypes = ["test"]
scale = "standardized_maxAbs" #"newBEST_Basic"
mask = tools.loadMask(maskPath)
dataDict = tools.loadH5Data(h5Dir, mask, sampleTypes, setTypes, scale)

model_BEST =  load_model("/uscms_data/d3/cannaert/BEST/CMSSW_10_6_27/src/BEST/training/trainingOutput/tweakedOldBEST/2018_BESvarList_tweakedOldBEST/BEST_model_2018_BESvarList_tweakedOldBEST.h5")
BESpredict = model_BEST.predict(dataDict["testEvents"][()])
plotDir = "/uscms_data/d3/cannaert/BEST/CMSSW_10_6_27/src/BEST/training/plots/tweakedOldBEST/testpt/"
suffix = ""#"standardized"
truth = np.argmax(dataDict["testTruth"], axis=1)


massptmask = tools.loadMask('/uscms_data/d3/cannaert/BEST/CMSSW_10_6_27/src/BEST/training/trainingOutput/tweakedOldBEST/2018_BESvarList_tweakedOldBEST/ptmask.txt', 82)
massptdict = tools.loadH5Data(h5Dir, massptmask, sampleTypes, setTypes, "")

tools.plotpTCM(BESpredict, truth, massptdict["testEvents"][:,0], plotDir, suffix)
# print("Loading h5 datasets...")
# myTrainEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened.h5",     "r")["BES_vars"])[:,:] for mySample in sampleTypes]
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print("Loading scaler model....")
# scaler = load('ScalerParameters/ScalerParameters_standard.joblib')
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print("Scaling Indv....")
# newEvents = [scaler.transform(arr) for arr in myTrainEvents]
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print("Cleaning Up....")
# del newEvents
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print("Concatenating....")
# events = np.concatenate(myTrainEvents)
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")


# print("Scaling All....")
# scaler.transform(events)
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print("Shuffling....")
# rng_state = np.random.get_state()
# np.random.set_state(rng_state)
# np.random.shuffle(events)
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print("Loading Mask....")
# maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask.txt"
# maskFile = open(maskPath, "r")
# maskIndex = []
# for line in maskFile:
#     maskIndex.append(line.split(':')[0])
# maskFile.close()

# print(maskPath + " chosen; mask size " + str(len(maskIndex)))
# myMask = [True if str(i) in maskIndex else False for i in range(596)]
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print("Loading h5 datasets...")
# myTrainEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened.h5",     "r")["BES_vars"])[:,myMask] for mySample in sampleTypes]
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# startTime = time.time()
# print("Loading h5 datasets...")
# myTrainEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened.h5",     "r")["BES_vars"])[()] for mySample in sampleTypes]
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# startTime = time.time()
# print("Concatenating....")
# globals()['events'] = np.concatenate(myTrainEvents)
# del myTrainEvents
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print(globals()['events'][0])

# startTime = time.time()
# print("Shuffling....")
# rng_state = np.random.get_state()
# np.random.set_state(rng_state)
# np.random.shuffle(globals()['events'])
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print(globals()['events'][0])

# do again but w/ functions

# setTypes = ["validation", "train"]
# setTypes = ["validation"]
# startTime = time.time()
# thisDict  = tools.loadH5Data(h5Dir, np.array([True,True]), sampleTypes, setTypes)
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print(thisDict.keys())
# print(thisDict["validationEvents"][0:10][0])
# print(thisDict["validationTruth"][0:10])

# startTime = time.time()
# rng_state = np.random.get_state()
# tools.shuffleArray(thisDict, rng_state)
# timeCheck = divmod(time.time() - startTime, 60.)
# print("Time Taken:" + str( int(timeCheck[0]) ) + "m " + str( int(timeCheck[1]) ) + "s")

# print(thisDict.keys())
# print(thisDict["validationEvents"][0:10][0])
# print(thisDict["validationTruth"][0:10])
# del thisDict
# del myTrainEvents



# # Check how long the script took to run
# timeTaken = divmod(time.time() - startTime, 60.)
# timeMessage = "Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
# print(timeMessage)
