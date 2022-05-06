import time
# startTime = time.time() # Tracks how long script takes

from sklearn import metrics, preprocessing



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


sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
h5Dir="/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
# maskPath = '/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_longLearn/300Basic_300_Z/newBESTMask_300_Z.txt'
maskPath = '/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_maskFix/longBasic_ak8/fixBESTMask_ak8.txt'
setTypes = ["test"]
scale = "newBEST_Basic"

# ptmask = tools.loadMask('/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_maskFix/longBasic_ak8/ptmask.txt')
# scaler = tools.loadScalerModel("",ptmask)
scaler = load("/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/newBEST_Basic.joblib")

ptdict = tools.loadH5Data(h5Dir, [True], sampleTypes, setTypes, scale)
# ptarray = ptdict["testEvents"][:,0]
ptarray = scaler.inverse_transform(ptdict["testEvents"][()])

with h5py.File("pt_unscaled_test.h5", "w") as fh5:
    fh5.create_dataset("pt", data=ptarray[:,518])
quit()

mask = tools.loadMask(maskPath)
dataDict = tools.loadH5Data(h5Dir, mask, sampleTypes, setTypes, scale)

# model_BEST = load_model('/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_longLearn/300Basic_300_Z/BEST_model_300Basic_300_Z.h5')
model_BEST = load_model('/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_maskFix/longBasic_ak8/BEST_model_longBasic_ak8.h5')
BESpredict = model_BEST.predict(dataDict["testEvents"][()])
# plotDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/plots/newBEST_longLearn/300Basic_300_Z/"
plotDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/plots/newBEST_maskFix/longBasic_ak8/"
# suffix = "300Basic"
suffix = "maskfix"
truth = np.argmax(dataDict["testTruth"], axis=1)


# ptmask = tools.loadMask('/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_maskFix/longBasic_ak8/ptmask.txt')
# scaler = tools.loadScalerModel("",ptmask)
# ptdict = tools.loadH5Data(h5Dir, ptmask, sampleTypes, setTypes, scale)
# ptarray = ptdict["testEvents"][:,0]
# ptarray = scaler.inverse_transform(ptarray)



tools.plotpTCM(BESpredict, truth, ptarray, plotDir, suffix)


# massptmask = tools.loadMask('/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_longLearn/300Basic_300_Z/masspT.txt', 596)
# massptdict = tools.loadH5Data(h5Dir, massptmask, sampleTypes, setTypes, "")

# tools.plotpTCM(BESpredict, truth, massptdict["testEvents"][:,0], plotDir, suffix)

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
