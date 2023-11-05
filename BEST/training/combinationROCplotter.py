#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# combonationROCplotter.py ////////////////////////////////////////////////////////
#==================================================================================
# This program evaluates BEST: HH Event Shape Topology Indentification Algorithm 
#==================================================================================

# modules
#import ROOT as root
import numpy
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import copy
import random

# get stuff from modules
#from root_numpy import tree2array
from scipy import interp
from sklearn import svm, metrics, preprocessing, neural_network, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.layers import Input, Activation, Dense, Conv2D, SeparableConv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, MaxoutDense
from keras.layers import concatenate
from keras.regularizers import l1,l2
from keras.utils import np_utils, to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# user modules
import tools.functions as tools

# enter batch mode in root (so python can access displays)
#root.gROOT.SetBatch(True)

# set options 
savePDF = False
savePNG = True 

setTypes = ["Test"]
sampleTypes = ["W","Z","Higgs","Top","b","QCD"]
frameTypes = ["W","Z","Higgs","Top"]

BESonly   = "/uscms/home/bonillaj/nobackup/models/BEST_model_BES.h5"
ImageOnly = "/uscms/home/bonillaj/nobackup/models/BEST_model_Images.h5"
#BESimage  = "/uscms_data/d3/bregnery/BEST/BESTdev/CMSSW_10_2_18/src/BEST/training/models/BEST_modeldeepSetLSTMCassini2017.h5"

BESimage  = "/uscms/home/bonillaj/nobackup/models/BEST_model_Both.h5"

#==================================================================================
# Load Test Data //////////////////////////////////////////////////////////////////
#==================================================================================

# This will create a series of global variables like jetTopFrameTrain and jetHiggsFrameValidation and jetBESvarsTrain, (4frames+1BesVars)*2sets=10globVars
for mySet in setTypes:
    for myFrame in frameTypes:
        globals()["jet"+myFrame+"Frame"+mySet] = []

    globals()["jetBESvars"+mySet] = []

    globals()["truthLabels"+mySet] = []

## and this makes 12 global variables to store data
print(globals().keys())

for mySet in setTypes:
    for index, mySample in enumerate(sampleTypes):
        print("Opening "+mySample+mySet+" file")
        myF = h5py.File("/uscms/home/bonillaj/nobackup/h5samples/"+mySample+"Sample_BESTinputs_"+mySet.lower()+"_flattened_standardized.h5","r")

        ## Make TruthLabels, only once (i.e. for key=BESvars)
        if globals()["truthLabels"+mySet] == []:
            print("Making new", "truthLabels"+mySet)
            globals()["truthLabels"+mySet] = numpy.full(len(myF['BES_vars'][()]), index)
        else:
            print("Concatenate", "truthLabels"+mySet)
            globals()["truthLabels"+mySet] = numpy.concatenate((globals()["truthLabels"+mySet], numpy.full(len(myF['BES_vars'][()]), index)))
         
        for myKey in myF.keys():
            varKey = "jet"
            if "image" in myKey.lower():
                varKey = varKey+myKey.split("_")[0] # so HiggsFrame, TopFrame, etc
            else:
                varKey = varKey+"BESvars"
            
            varKey = varKey+mySet
      
            ## Append data
            if globals()[varKey] == []:
                print("Making new", varKey)
                globals()[varKey] = myF[myKey][()]
            else:
                print("Concatenate", varKey)
                globals()[varKey] = numpy.concatenate((globals()[varKey], myF[myKey][()]))
         
        myF.close()
print("Successfully loaded data from h5 files")

#make the truth categories appropriately
print("To_Categorical")
for mySet in setTypes:
    globals()["truthLabels"+mySet] = to_categorical(globals()["truthLabels"+mySet], num_classes = 6)
    print("Made Truth Labels "+mySet, globals()["truthLabels"+mySet].shape)

#==================================================================================
# Load BES only network ///////////////////////////////////////////////////////////
#==================================================================================

model_BESonly = load_model(BESonly)

print("Loaded the BES variable only neural network!")

#==================================================================================
# Load image only network /////////////////////////////////////////////////////////
#==================================================================================

#model_imageOnly = load_model(ImageOnly)
model_imageOnly = load_model(BESonly)

print("Loaded the boosted jet image only neural network!")

#==================================================================================
# Load the BEST network ///////////////////////////////////////////////////////////
#==================================================================================

model_BEST = load_model(BESimage)

print("Loaded the BEST neural network!")

#==================================================================================
# Load the BEST PCT results ///////////////////////////////////////////////////////
#==================================================================================

file_PCT = h5py.File('/uscms/home/bonillaj/nobackup/Brendan/CMSSW_10_2_18/src/PCT_HEP/logs/log/BESTOUTPUT.h5', 'r')
PCTpredict = file_PCT['DNN']
PCTtruth = to_categorical(file_PCT['pid'], num_classes=6)

#==================================================================================
# Create ROC Curves ///////////////////////////////////////////////////////////////
#==================================================================================

# predict
BESpredict = model_BESonly.predict([globals()["jetBESvarsTest"][:]])
BJIpredict = model_imageOnly.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:]])
BESTpredict = model_BEST.predict([globals()["jetWFrameTest"][:], globals()["jetZFrameTest"][:], globals()["jetHiggsFrameTest"][:], globals()["jetTopFrameTest"][:], globals()["jetBESvarsTest"][:] ])


print("Made predictions using the neural networks")

# BEST
# compute ROC curve and area for each class
n_classes = globals()["truthLabelsTest"].shape[1] 
fprBEST = dict()
tprBEST = dict()
roc_auc_BEST = dict()
for i in range(n_classes):
    fprBEST[i], tprBEST[i], _ = roc_curve(globals()["truthLabelsTest"][:, i], BESTpredict[:, i]) # returns 3 outputs but only care about 2
    roc_auc_BEST[i] = auc(fprBEST[i], tprBEST[i])

# compute micro-average ROC curve and ROC area
fprBEST["micro"], tprBEST["micro"], _ = roc_curve(globals()["truthLabelsTest"].ravel(), BESTpredict.ravel() )
roc_auc_BEST["micro"] = auc(fprBEST["micro"], tprBEST["micro"] )

# compute macro-average ROC curve and ROC area
# first aggregate all false positive rates
all_fprBEST = numpy.unique(numpy.concatenate([fprBEST[i] for i in range(n_classes)]) )

# interpolate all roc curves
mean_tprBEST = numpy.zeros_like(all_fprBEST)
for i in range(n_classes):
    mean_tprBEST += interp(all_fprBEST, fprBEST[i], tprBEST[i] )

# average and compute macro AUC
mean_tprBEST /= n_classes

fprBEST["macro"] = all_fprBEST
tprBEST["macro"] = mean_tprBEST
roc_auc_BEST["macro"] = auc(fprBEST["macro"], tprBEST["macro"])

# BES only
# compute ROC curve and area for each class
n_classes = globals()["truthLabelsTest"].shape[1] 
fprBES = dict()
tprBES = dict()
roc_auc_BES = dict()
for i in range(n_classes):
    fprBES[i], tprBES[i], _ = roc_curve(globals()["truthLabelsTest"][:, i], BESpredict[:, i])
    roc_auc_BES[i] = auc(fprBES[i], tprBES[i])

# compute micro-average ROC curve and ROC area
fprBES["micro"], tprBES["micro"], _ = roc_curve(globals()["truthLabelsTest"].ravel(), BESpredict.ravel() )
roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

# compute macro-average ROC curve and ROC area
# first aggregate all false positive rates
all_fprBES = numpy.unique(numpy.concatenate([fprBES[i] for i in range(n_classes)]) )

# interpolate all roc curves
mean_tprBES = numpy.zeros_like(all_fprBES)
for i in range(n_classes):
    mean_tprBES += interp(all_fprBES, fprBES[i], tprBES[i] )

# average and compute macro AUC
mean_tprBES /= n_classes

fprBES["macro"] = all_fprBES
tprBES["macro"] = mean_tprBES
roc_auc_BES["macro"] = auc(fprBES["macro"], tprBES["macro"])

# Boosted Jet Images only
# compute ROC curve and area for each class
n_classes = globals()["truthLabelsTest"].shape[1]
fprBJI = dict()
tprBJI = dict()
roc_auc_BJI = dict()
for i in range(n_classes):
    fprBJI[i], tprBJI[i], _ = roc_curve(globals()["truthLabelsTest"][:, i], BJIpredict[:, i]) # returns 3 outputs but only care about 2
    roc_auc_BJI[i] = auc(fprBJI[i], tprBJI[i])

# compute micro-average ROC curve and ROC area
fprBJI["micro"], tprBJI["micro"], _ = roc_curve(globals()["truthLabelsTest"].ravel(), BJIpredict.ravel() )
roc_auc_BJI["micro"] = auc(fprBJI["micro"], tprBJI["micro"] )

# compute macro-average ROC curve and ROC area
# first aggregate all false positive rates
all_fprBJI = numpy.unique(numpy.concatenate([fprBJI[i] for i in range(n_classes)]) )

# interpolate all roc curves
mean_tprBJI = numpy.zeros_like(all_fprBJI)
for i in range(n_classes):
    mean_tprBJI += interp(all_fprBJI, fprBJI[i], tprBJI[i] )

# average and compute macro AUC
mean_tprBJI /= n_classes

fprBJI["macro"] = all_fprBJI
tprBJI["macro"] = mean_tprBJI
roc_auc_BJI["macro"] = auc(fprBJI["macro"], tprBJI["macro"])

print("Created ROC curves")

# PCT network 
# compute ROC curve and area for each class
n_classes = globals()["truthLabelsTest"].shape[1]
fprPCT = dict()
tprPCT = dict()
roc_auc_PCT = dict()
for i in range(n_classes):
    fprPCT[i], tprPCT[i], _ = roc_curve(PCTtruth[:, i], PCTpredict[:, i]) # returns 3 outputs but only care about 2
    roc_auc_PCT[i] = auc(fprPCT[i], tprPCT[i])

# compute micro-average ROC curve and ROC area
fprPCT["micro"], tprPCT["micro"], _ = roc_curve(globals()["truthLabelsTest"].ravel(), PCTpredict.ravel() )
roc_auc_PCT["micro"] = auc(fprPCT["micro"], tprPCT["micro"] )

# compute macro-average ROC curve and ROC area
# first aggregate all false positive rates
all_fprPCT = numpy.unique(numpy.concatenate([fprPCT[i] for i in range(n_classes)]) )

# interpolate all roc curves
mean_tprPCT = numpy.zeros_like(all_fprPCT)
for i in range(n_classes):
    mean_tprPCT += interp(all_fprPCT, fprPCT[i], tprPCT[i] )

# average and compute macro AUC
mean_tprPCT /= n_classes

fprPCT["macro"] = all_fprPCT
tprPCT["macro"] = mean_tprPCT
roc_auc_PCT["macro"] = auc(fprPCT["macro"], tprPCT["macro"])

print("Created ROC curves")

#==================================================================================
# Plot ROC curves /////////////////////////////////////////////////////////////////
#==================================================================================

# average ROC
plt.figure(1)
plt.plot(fprBES["micro"], tprBES["micro"],
         label='BES micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES["micro"]),
         color='orange', linewidth=2)
plt.plot(fprBJI["micro"], tprBJI["micro"],
         label='BJI micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI["micro"]),
         color='deeppink', linewidth=2)
plt.plot(fprBEST["micro"], tprBEST["micro"],
         label='BEST micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST["micro"]),
         color='blue', linewidth=2)
plt.plot(fprPCT["micro"], tprPCT["micro"],
         label='PCT micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT["micro"]),
         color='green', linewidth=2)
plt.plot(fprBES["macro"], tprBES["macro"],
         label='BES macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES["macro"]),
         color='orange', linestyle=':', linewidth=4)
plt.plot(fprBJI["macro"], tprBJI["macro"],
         label='BJI macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI["macro"]),
         color='deeppink', linestyle=':', linewidth=4)
plt.plot(fprBEST["macro"], tprBEST["macro"],
         label='BEST macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST["macro"]),
         color='blue', linestyle=':', linewidth=4)
plt.plot(fprPCT["macro"], tprPCT["macro"],
         label='PCT macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT["macro"]),
         color='green', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_average_ROCplot.png')
plt.close()

# category ROC curves
# W boson
plt.figure(1)
plt.plot(fprBES[0], tprBES[0],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[0]),
         color='orange', linewidth=2)
plt.plot(fprBJI[0], tprBJI[0],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[0]),
         color='deeppink', linewidth=2)
plt.plot(fprBEST[0], tprBEST[0],
         label='BEST ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST[0]),
         color='blue', linewidth=2)
plt.plot(fprPCT[0], tprPCT[0],
         label='PCT ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT[0]),
         color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('W category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_W_ROCplot.png')
plt.close()

# Z boson
plt.figure(1)
plt.plot(fprBES[1], tprBES[1],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[1]),
         color='orange', linewidth=2)
plt.plot(fprBJI[1], tprBJI[1],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[1]),
         color='deeppink', linewidth=2)
plt.plot(fprBEST[1], tprBEST[1],
         label='BEST ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST[1]),
         color='blue', linewidth=2)
plt.plot(fprPCT[1], tprPCT[1],
         label='PCT ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT[1]),
         color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Z category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_Z_ROCplot.png')
plt.close()

# Higgs
plt.figure(1)
plt.plot(fprBES[2], tprBES[2],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[2]),
         color='orange', linewidth=2)
plt.plot(fprBJI[2], tprBJI[2],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[2]),
         color='deeppink', linewidth=2)
plt.plot(fprBEST[2], tprBEST[2],
         label='BEST ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST[2]),
         color='blue', linewidth=2)
plt.plot(fprPCT[2], tprPCT[2],
         label='PCT ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT[2]),
         color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Higgs category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_H_ROCplot.png')
plt.close()

# Top
plt.figure(1)
plt.plot(fprBES[3], tprBES[3],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[3]),
         color='orange', linewidth=2)
plt.plot(fprBJI[3], tprBJI[3],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[3]),
         color='deeppink', linewidth=2)
plt.plot(fprBEST[3], tprBEST[3],
         label='BEST ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST[3]),
         color='blue', linewidth=2)
plt.plot(fprPCT[3], tprPCT[3],
         label='PCT ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT[3]),
         color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Top category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_t_ROCplot.png')
plt.close()

# bottom
plt.figure(1)
plt.plot(fprBES[4], tprBES[4],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[4]),
         color='orange', linewidth=2)
plt.plot(fprBJI[4], tprBJI[4],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[4]),
         color='deeppink', linewidth=2)
plt.plot(fprBEST[4], tprBEST[4],
         label='BEST ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST[4]),
         color='blue', linewidth=2)
plt.plot(fprPCT[4], tprPCT[4],
         label='PCT ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT[4]),
         color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Bottom category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_b_ROCplot.png')
plt.close()

# QCD
plt.figure(1)
plt.plot(fprBES[5], tprBES[5],
         label='BES only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BES[5]),
         color='orange', linewidth=2)
plt.plot(fprBJI[5], tprBJI[5],
         label='Image only ROC curve (area = {0:0.2f})' ''.format(roc_auc_BJI[5]),
         color='deeppink', linewidth=2)
plt.plot(fprBEST[5], tprBEST[5],
         label='BEST ROC curve (area = {0:0.2f})' ''.format(roc_auc_BEST[5]),
         color='blue', linewidth=2)
plt.plot(fprPCT[5], tprPCT[5],
         label='PCT ROC curve (area = {0:0.2f})' ''.format(roc_auc_PCT[5]),
         color='green', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('QCD category ROC Curves')
plt.legend(loc="lower right")
plt.savefig('plots/comparison_QCD_ROCplot.png')
plt.close()


print("Plotted ROC curves")



