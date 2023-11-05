#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# johanTraining.py ////////////////////////////////////////////////////////////////
#==================================================================================
# This program trains BEST with flattened inputs //////////////////////////////////
# One can train the network with only BESvars and images, or both /////////////////
# One can also ask to run the ensemble which takes/creates BES-only and ///////////
# image-only networks, and feeds the output predictions into a separate network ///
#==================================================================================

# modules
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
import numpy.random
import math

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
config.gpu_options.per_process_gpu_memory_fraction = 0.7
k.tensorflow_backend.set_session(tf.Session(config=config))

# user modules
import tools.functions as tools
print("I am I really")
from plotConfusionMatrix import makeCM
print("Getting stuck?")

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

# Do BES and/or images
doBES = False
doImages = False
doEnsemble = False
mySuffix = ""

sampleTypes = ["Zt","Ht","QCD1500to2000","QCD2000toInf","Wb","TTBar"]
frameTypes = [""]

BatchSize = 1200

def loadData(h5Dir, year, setTypes):
   return
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

   #print(globals().keys())

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
            print("Shape I care:", numpy.array(globals()["truthLabels"+mySet]).shape, numpy.array(globals()["truthLabels"+mySet][0:100]).shape)
            globals()["truthLabels"+mySet] = numpy.full(len(myF['BES_vars'][()]), index)
         else:
            print("Concatenate", "truthLabels"+mySet)
            print("Shape I care:", globals()["truthLabels"+mySet].shape, globals()["truthLabels"+mySet][0:100].shape)
            globals()["truthLabels"+mySet] = numpy.concatenate((globals()["truthLabels"+mySet], numpy.full(len(myF['BES_vars'][()]), index)))
            
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
               print("Shape I care:", numpy.array(myF[myKey][()].shape), numpy.array(myF[myKey][()][0:100]).shape)
               globals()[varKey] = myF[myKey][()]
            else:
               print("Concatenate", varKey)
               globals()[varKey] = numpy.concatenate((globals()[varKey], myF[myKey][()]))
            
         myF.close()
         
   print("Finished Accessing H5 data")
   ## Order of categories: 0-W, 1-Z, 2-H, 3-t, 4-b, 5-QCD (order of sampleTypes). Format properly.
   print("To_Categorical")
   for mySet in setTypes:
      globals()["truthLabels"+mySet] = to_categorical(globals()["truthLabels"+mySet], num_classes = 6)
      print("Made Truth Labels "+mySet, globals()["truthLabels"+mySet].shape)


def train(doBES, doImages, h5Dir, outDir, suffix, userPatience):
   """
   if doBES:
      print("BESvars Train Shape", globals()["jetBESvarsTrain"].shape)
      print("BESvars Validation Shape", globals()["jetBESvarsValidation"].shape)
   if doImages:
      for myFrame in frameTypes:
         print(myFrame+" Images Train Shape", globals()["jet"+myFrame+"FrameTrain"].shape)
         print(myFrame+" Images Validation Shape", globals()["jet"+myFrame+"FrameValidation"].shape)

   
   print("Shuffle Train")
   rng_state = numpy.random.get_state()
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(globals()["truthLabelsTrain"])
   if doBES:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsTrain"])
   if doImages:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameTrain"])

   print("Shuffle Validation")
   numpy.random.set_state(rng_state)
   numpy.random.shuffle(globals()["truthLabelsValidation"])
   if doBES:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsValidation"])
   if doImages:
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameValidation"])

   
   print("Stored data and truth information")
   """
   #==================================================================================
   # Train the Neural Network ////////////////////////////////////////////////////////
   #==================================================================================
   # Shape parameters
   if doImages:
      arbitrary_length = 10 #Hopefully this number doesn't matter
      nx = 31
      ny = 31
      """
      ImageShapeHolder = numpy.zeros((arbitrary_length, nx, ny, 4))

      ImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )
   
      ImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageInputs)
      ImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageLayer)
      ImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageLayer)
      ImageLayer = BatchNormalization(momentum = 0.6)(ImageLayer)
      ImageLayer = MaxPool2D(pool_size=(2,2) )(ImageLayer)
      ImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageLayer)
      ImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageLayer)
      ImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageLayer)
      ImageLayer = BatchNormalization(momentum = 0.6)(ImageLayer)
      ImageLayer = MaxPool2D(pool_size=(2,2) )(ImageLayer)
      ImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageLayer)
      ImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ImageLayer)
      ImageLayer = BatchNormalization(momentum = 0.6)(ImageLayer)
      ImageLayer = MaxPool2D(pool_size=(2,2) )(ImageLayer) 
      ImageLayer = Flatten()(ImageLayer)
      ImageLayer = Dropout(0.20)(ImageLayer)
      #HiggsImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
      #HiggsImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
      #HiggsImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
      #HiggsImageLayer = Dropout(0.10)(HiggsImageLayer)
   
      ImageModel = Model(inputs = ImageInputs, outputs = ImageLayer)

      """
      ImageShapeHolder = numpy.zeros((arbitrary_length, nx, ny, 1))

      HiggsImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )
   
      HiggsImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageInputs)
      HiggsImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
      HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
      HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
      HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer)
      HiggsImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
      HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
      HiggsImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
      HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
      HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer)
      HiggsImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
      HiggsImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(HiggsImageLayer)
      HiggsImageLayer = BatchNormalization(momentum = 0.6)(HiggsImageLayer)
      HiggsImageLayer = MaxPool2D(pool_size=(2,2) )(HiggsImageLayer)
      HiggsImageLayer = Flatten()(HiggsImageLayer)
      HiggsImageLayer = Dropout(0.20)(HiggsImageLayer)
      #HiggsImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
      #HiggsImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
      #HiggsImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(HiggsImageLayer)
      #HiggsImageLayer = Dropout(0.10)(HiggsImageLayer)
   
      HiggsImageModel = Model(inputs = HiggsImageInputs, outputs = HiggsImageLayer)
   
      #Top image
      TopImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )
      
      TopImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageInputs)
      TopImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
      TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
      TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
      TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
      TopImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
      TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
      TopImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
      TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
      TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
      TopImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
      TopImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(TopImageLayer)
      TopImageLayer = BatchNormalization(momentum = 0.6)(TopImageLayer)
      TopImageLayer = MaxPool2D(pool_size=(2,2) )(TopImageLayer)
      TopImageLayer = Flatten()(TopImageLayer)
      TopImageLayer = Dropout(0.20)(TopImageLayer)
      #TopImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
      #TopImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
      #TopImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(TopImageLayer)
      #TopImageLayer = Dropout(0.10)(TopImageLayer)
   
      TopImageModel = Model(inputs = TopImageInputs, outputs = TopImageLayer)
   
      #W Model
      WImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )
      
      WImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageInputs)
      WImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
      WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
      WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
      WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
      WImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
      WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
      WImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
      WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
      WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
      WImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
      WImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(WImageLayer)
      WImageLayer = BatchNormalization(momentum = 0.6)(WImageLayer)
      WImageLayer = MaxPool2D(pool_size=(2,2) )(WImageLayer)
      WImageLayer = Flatten()(WImageLayer)
      WImageLayer = Dropout(0.20)(WImageLayer)
      #WImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
      #WImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
      #WImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(WImageLayer)
      #WImageLayer = Dropout(0.10)(WImageLayer)

      WImageModel = Model(inputs = WImageInputs, outputs = WImageLayer)
      
      
      #Z Model
      ZImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )
      
      ZImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageInputs)
      ZImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
      ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
      ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
      ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
      ZImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
      ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
      ZImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
      ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
      ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
      ZImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
      ZImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(ZImageLayer)
      ZImageLayer = BatchNormalization(momentum = 0.6)(ZImageLayer)
      ZImageLayer = MaxPool2D(pool_size=(2,2) )(ZImageLayer)
      ZImageLayer = Flatten()(ZImageLayer)
      ZImageLayer = Dropout(0.20)(ZImageLayer)#try 0.35 dropout here and in other images networks
      #ZImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
      #ZImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
      #ZImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(ZImageLayer)
      #ZImageLayer = Dropout(0.10)(ZImageLayer)

      ZImageModel = Model(inputs = ZImageInputs, outputs = ZImageLayer)
      
      #Bottom Model
      BottomImageInputs = Input( shape=(ImageShapeHolder.shape[1], ImageShapeHolder.shape[2], ImageShapeHolder.shape[3]) )
      
      BottomImageLayer = SeparableConv2D(32, (11,11), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageInputs)
      BottomImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageLayer)
      BottomImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageLayer)
      BottomImageLayer = BatchNormalization(momentum = 0.6)(BottomImageLayer)
      BottomImageLayer = MaxPool2D(pool_size=(2,2) )(BottomImageLayer)
      BottomImageLayer = SeparableConv2D(32, (7,7), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageLayer)
      BottomImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageLayer)
      BottomImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageLayer)
      BottomImageLayer = BatchNormalization(momentum = 0.6)(BottomImageLayer)
      BottomImageLayer = MaxPool2D(pool_size=(2,2) )(BottomImageLayer)
      BottomImageLayer = SeparableConv2D(32, (5,5), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageLayer)
      BottomImageLayer = SeparableConv2D(32, (2,2), strides=(1,1), padding="same", activation="relu", kernel_regularizer=l2(0.01) )(BottomImageLayer)
      BottomImageLayer = BatchNormalization(momentum = 0.6)(BottomImageLayer)
      BottomImageLayer = MaxPool2D(pool_size=(2,2) )(BottomImageLayer)
      BottomImageLayer = Flatten()(BottomImageLayer)
      BottomImageLayer = Dropout(0.20)(BottomImageLayer)#try 0.35 dropout here and in other images networks
      #BottomImageLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(BottomImageLayer)
      #BottomImageLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(BottomImageLayer)
      #BottomImageLayer = Dense(24, kernel_initializer="glorot_normal", activation="relu" )(BottomImageLayer)
      #BottomImageLayer = Dropout(0.10)(BottomImageLayer)

      BottomImageModel = Model(inputs = BottomImageInputs, outputs = BottomImageLayer)
      

   # Create the BES variable version
   if doBES:
      if "oldBEST" in suffix:
         BestShapeHolder = 59
      else:
         BestShapeHolder = 123
      besInputs = Input( shape=(BestShapeHolder, ) )
      
      #besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besInputs)
      #besLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(besLayer)
         
      besModel = Model(inputs = besInputs, outputs = besInputs)
      print (besModel.output)   

   # Add BES variables to the network
   if doBES and not doImages:
      combined = besModel.output
   elif not doBES and doImages:
      combined = concatenate([WImageModel.output, ZImageModel.output, HiggsImageModel.output, TopImageModel.output, BottomImageModel.output])
      #combined = concatenate([WImageModel.output, ZImageModel.output, HiggsImageModel.output, TopImageModel.output])
      #combined = ImageModel.output
   elif doBES and doImages:
      combined = concatenate([WImageModel.output, ZImageModel.output, HiggsImageModel.output, TopImageModel.output, BottomImageModel.output, besModel.output])

   if "oldBEST" in suffix:
      #The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
      combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combined)
      combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
      combLayer = Dense(40, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
      outputModel = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)
   else:
      combLayer = Dense(512, kernel_initializer="glorot_normal", activation="relu" )(combined)
      combLayer = Dropout(0.35)(combLayer)# try 0.35
      combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
      #Another dropout of 0.35
      combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
      combLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
      combLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(combLayer)
      combLayer = Dropout(0.35)(combLayer)
      outputModel = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

   # compile the model
   if doBES and not doImages:
      myModel = Model(inputs = [besModel.input], outputs = outputModel)
   elif not doBES and doImages:
      myModel = Model(inputs = [WImageModel.input, ZImageModel.input, HiggsImageModel.input, TopImageModel.input, BottomImageModel.input], outputs = outputModel)
      #myModel = Model(inputs = [WImageModel.input, ZImageModel.input, HiggsImageModel.input, TopImageModel.input], outputs = outputModel)
      #myModel = Model(inputs = [ImageModel.input], outputs = outputModel)
   elif doBES and doImages:
      print("myModel with images and BESvars")
      myModel = Model(inputs = [WImageModel.input, ZImageModel.input, HiggsImageModel.input, TopImageModel.input, BottomImageModel.input, besModel.input], outputs = outputModel)

   myModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   
   print(myModel.summary() )

   # early stopping
   early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=userPatience, verbose=0, mode='auto')#, restore_best_weights=True,)


   # model checkpoint callback
   # this saves the model architecture + parameters into dense_model.h5
   
   model_checkpoint = ModelCheckpoint(outDir+'BEST_model'+suffix+'.h5', monitor='val_loss', 
                                      verbose=0, save_best_only=True, 
                                      save_weights_only=False, mode='auto', 
                                      period=1)

   # train the neural network
   if doBES and not doImages:
      #myTrainFilePath = outDir+"FinalSampleCompressedAllImages_2017_train_flattened_standardized_shuffled_v4.h5"
      #print("Before Open File",myTrainFilePath)
      #myTrainFile = h5py.File(myTrainFilePath,"r")
      #myValidationFilePath = outDir+"FinalSampleCompressedAllImages_2017_validation_flattened_standardized_shuffled_v4.h5"
      #print("Before Open Second File",myValidationFilePath)
      #myValidationFile = h5py.File(myValidationFilePath,"r")
      #print("Opened Files")

      if 'oldBEST' in suffix:
         maskFile = open("/uscms/home/msabbott/nobackup/abbott/CMSSW_10_6_27/src/BEST/formatConverter/masks/oldBESTMask.txt", "r")
         oldBESTMaskIndex = []
         for line in maskFile:
            oldBESTMaskIndex.append(line.split(':')[0])
         maskFile.close()
         print("Old BEST Mask size", len(oldBESTMaskIndex))
         myMask = [True if str(i) in oldBESTMaskIndex else False for i in range(596)]
      else:
         myMask = [True for i in range(123)]
         
      myTrainEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["BES_vars"])[:,myMask] for mySample in sampleTypes]
      myValidationEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["BES_vars"])[:,myMask] for mySample in sampleTypes]
      print("My Train events shape:",[myTrainEvents[i].shape for i in range(len(myTrainEvents))])
      print("My Validation events shape:",[myValidationEvents[i].shape for i in range(len(myValidationEvents))])
      myTrainTruth = [numpy.zeros((len(myTrainEvents[i]),len(sampleTypes))) for i in range(len(myTrainEvents))] # shape: N,6 filled w/ 0s
      myValidationTruth = [numpy.zeros((len(myValidationEvents[i]),len(sampleTypes))) for i in range(len(myValidationEvents))]
      for i in range(len(sampleTypes)):
         myTrainTruth[i][:,i] = 1.
         myValidationTruth[i][:,i] = 1.
      print("My train truth shape:",[myTrainTruth[i].shape for i in range(len(myTrainTruth))])
      print("My validation truth shape:",[myValidationTruth[i].shape for i in range(len(myValidationTruth))]) 
      print("Labels are:",[[i,mySample] for i,mySample in enumerate(sampleTypes)])
      globals()["jetBESvarsTrain"] = numpy.concatenate(myTrainEvents)
      globals()["truthLabelsTrain"] = numpy.concatenate(myTrainTruth)
      globals()["jetBESvarsValidation"] = numpy.concatenate(myValidationEvents)
      globals()["truthLabelsValidation"] = numpy.concatenate(myValidationTruth)
      print("Globals train shapes",globals()["jetBESvarsTrain"].shape, globals()["truthLabelsTrain"].shape, globals()["truthLabelsTrain"][0])
      print("Globals validation shapes",globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
         
      print("Shuffle Train")
      rng_state = numpy.random.get_state()
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["truthLabelsTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsTrain"])
      
      print("Shuffle Validation")
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["truthLabelsValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsValidation"])
      
      print("Globals train shapes",globals()["jetBESvarsTrain"].shape, globals()["truthLabelsTrain"].shape, globals()["truthLabelsTrain"][0])
      print("Globals validation shapes",globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
      #quit()
      print("Batch Size: "+str(BatchSize)+", Epochs: 50")
      if 'oldBEST' in suffix:
         history = myModel.fit([globals()["jetBESvarsTrain"][0:500000] ], globals()["truthLabelsTrain"][0:500000], batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]])
        #  history = myModel.fit([globals()["jetBESvarsTrain"][:] ], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]])
      else:
         history = myModel.fit([globals()["jetBESvarsTrain"][:] ], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]])
      
      """
      else:
         import trainGenerator
      
         myTrainFilePath = outDir+"FinalSampleCompressedAllImages_2017_train_flattened_standardized_shuffled_v4.h5"
         myTrainSize = 2070326
         train_generator = trainGenerator.DataGenerator("train", myTrainFilePath, myTrainSize, batch_size=BatchSize)
         
         myValidationFilePath = outDir+"FinalSampleCompressedAllImages_2017_validation_flattened_standardized_shuffled_v4.h5"
         #myValidationSize = 257363
         #validation_generator = trainGenerator.DataGenerator("validation", myValidationFilePath, myValidationSize, batch_size=BatchSize)
         
         myValidationFile = h5py.File(myValidationFilePath,"r")
         
         #history = myModel.fit(generator=train_generator,
         history = myModel.fit_generator(generator=train_generator,
                                         steps_per_epoch=800,
                                         epochs=50,
                                         #validation_data=validation_generator#,
                                         validation_data=[[myValidationFile["jetBESvarsvalidation"][:]], myValidationFile["truthLabelsvalidation"][:]],
                                         #batch_size=BatchSize,
                                         #callbacks=[model_checkpoint]#,
                                         #use_multiprocessing=False,
                                         workers=6
         )

         #train_generator, steps_per_epoch=int(myTrainSize/BatchSize), epochs=50, validation_data=validation_generator, validation_steps=int(myValidationSize/BatchSize))#, use_multiprocessing=True, workers=6)#validation_data=[[myValidationFile["jetBESvarsvalidation"][:]], myValidationFile["truthLabelsvalidation"][:]], use_multiprocessing=True, workers=6)#
      
      #history = myModel.fit([myTrainFile["jetBESvarstrain"][1000000:1500000]], myTrainFile["truthLabelstrain"][1000000:1500000], batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint], validation_data = [[myValidationFile["jetBESvarsvalidation"][0:10000]], myValidationFile["truthLabelsvalidation"][0:10000]])
      #myTrainFile.close()
      #myValidationFile.close()
      #history = myModel.fit([globals()["jetBESvarsTrain"][:] ], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=200, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]])
      """

   elif not doBES and doImages:
      print("Shapes!")
      """
      print("Train Images shape for single frame:",globals()["jetWFrameTrain"].shape)
      concatStepTrain = int(math.ceil(globals()["jetWFrameTrain"].shape[0]/10.))
      print("ConcatStepTrain:",concatStepTrain)
      trainImages = []#numpy.zeros((globals()["jetWFrameTrain"].shape[0],31,31,4), dtype=float)
      for i in range(0,10):
         print("Train Step",i)
         eventLowTrain = i*concatStepTrain
         eventHighTrain = min(globals()["jetWFrameTrain"].shape[0],(i+1)*concatStepTrain)
         print("Events:",eventLowTrain,eventHighTrain)
         if i == 0:
            trainImages = tf.keras.layers.Concatenate(axis=3)([globals()["jetWFrameTrain"][eventLowTrain:eventHighTrain], globals()["jetZFrameTrain"][eventLowTrain:eventHighTrain], globals()["jetHiggsFrameTrain"][eventLowTrain:eventHighTrain], globals()["jetTopFrameTrain"][eventLowTrain:eventHighTrain]])[()]
         else:
            newSet = tf.keras.layers.Concatenate(axis=3)([globals()["jetWFrameTrain"][eventLowTrain:eventHighTrain], globals()["jetZFrameTrain"][eventLowTrain:eventHighTrain], globals()["jetHiggsFrameTrain"][eventLowTrain:eventHighTrain], globals()["jetTopFrameTrain"][eventLowTrain:eventHighTrain]])[()]
            print("Sizes:",trainImages.shape,newSet.shape)
            trainImages = tf.keras.layers.Concatenate(axis=0)([trainImages[:], newSet[:]])[()]
      print("After concat shape train", trainImages.shape)
      for myFrame in frameTypes:
         if "jet"+myFrame+"FrameTrain" in globals().keys():
            del globals()["jet"+myFrame+"FrameTrain"]
      
      print("Validation Images shape for single frame:",globals()["jetWFrameValidation"].shape)
      concatStepValidation = int(math.ceil(globals()["jetWFrameValidation"].shape[0]/10.))
      print("ConcatStepValidation:",concatStepValidation)
      validationImages = []#numpy.zeros((globals()["jetWFrameValidation"].shape[0],31,31,4), dtype=float)
      for i in range(0,10):
         print("Validation Step",i)
         eventLowValidation = i*concatStepValidation
         eventHighValidation = min(globals()["jetWFrameValidation"].shape[0],(i+1)*concatStepValidation)
         print("Validation Events:",eventLowValidation,eventHighValidation)
         if i == 0:
            validationImages = tf.keras.layers.Concatenate(axis=3)([globals()["jetWFrameValidation"][eventLowValidation:eventHighValidation], globals()["jetZFrameValidation"][eventLowValidation:eventHighValidation], globals()["jetHiggsFrameValidation"][eventLowValidation:eventHighValidation], globals()["jetTopFrameValidation"][eventLowValidation:eventHighValidation]])
         else:
            newSet = tf.keras.layers.Concatenate(axis=3)([globals()["jetWFrameValidation"][eventLowValidation:eventHighValidation], globals()["jetZFrameValidation"][eventLowValidation:eventHighValidation], globals()["jetHiggsFrameValidation"][eventLowValidation:eventHighValidation], globals()["jetTopFrameValidation"][eventLowValidation:eventHighValidation]])[()]
            print("Sizes:",validationImages.shape,newSet.shape)
            validationImages = tf.keras.layers.Concatenate(axis=0)([validationImages[:], newSet[:]])[()]
      print("After concat shape validation", validationImages.shape)
      for myFrame in frameTypes:
         if "jet"+myFrame+"FrameValidation" in globals().keys():
            del globals()["jet"+myFrame+"FrameValidation"]
      """
      print("Done")
      #history = myModel.fit(trainImages, globals()["truthLabelsTrain"][:], steps_per_epoch=math.ceil(int(trainImages.shape[0])/BatchSize), epochs=50, callbacks=[early_stopping, model_checkpoint], validation_data = [validationImages, globals()["truthLabelsValidation"][:]], validation_steps=math.ceil(int(validationImages.shape[0])/BatchSize))
      #history = myModel.fit([globals()["jetWFrameTrain"][:], globals()["jetZFrameTrain"][:], globals()["jetHiggsFrameTrain"][:], globals()["jetTopFrameTrain"][:]], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=100, callbacks=[early_stopping, model_checkpoint], validation_data = [[globals()["jetWFrameValidation"][:], globals()["jetZFrameValidation"][:], globals()["jetHiggsFrameValidation"][:], globals()["jetTopFrameValidation"][:]], globals()["truthLabelsValidation"][:]])
      #print("Before Open File")
      #myTrainFile = h5py.File(outDir+"FinalSampleAllImages_2017_train_flattened_standardized_shuffled.h5","r")
      #print("Before Open Second File")
      #myValidationFile = h5py.File(outDir+"FinalSampleAllImages_2017_validation_flattened_standardized_shuffled.h5","r")
      #print("Opened Files")
      #history = myModel.fit([myTrainFile["jetWFrametrain"][:], myTrainFile["jetZFrametrain"][:], myTrainFile["jetHiggsFrametrain"][:], myTrainFile["jetTopFrametrain"][:], myTrainFile["jetBottomFrametrain"][:]], myTrainFile["truthLabelstrain"][:], batch_size=BatchSize, epochs=100, callbacks=[early_stopping, model_checkpoint], validation_data = [[myValidationFile["jetWFramevalidation"][:], myValidationFile["jetZFramevalidation"][:], myValidationFile["jetHiggsFramevalidation"][:], myValidationFile["jetTopFramevalidation"][:], myValidationFile["jetBottomFramevalidation"][:]], myValidationFile["truthLabelsvalidation"][:]])
      #history = myModel.fit([myTrainFile["jetWFrametrain"][:], myTrainFile["jetZFrametrain"][:], myTrainFile["jetHiggsFrametrain"][:], myTrainFile["jetTopFrametrain"][:]], myTrainFile["truthLabelstrain"][:], batch_size=BatchSize, epochs=100, callbacks=[early_stopping, model_checkpoint], validation_data = [[myValidationFile["jetWFramevalidation"][:], myValidationFile["jetZFramevalidation"][:], myValidationFile["jetHiggsFramevalidation"][:], myValidationFile["jetTopFramevalidation"][:]], myValidationFile["truthLabelsvalidation"][:]])
      #myTrainFile.close()
      #myValidationFile.close()

      """
      import trainGenerator
      
      myTrainFilePath = outDir+"FinalSampleCompressedAllImages_2017_train_flattened_standardized_shuffled_v4.h5"
      myTrainSize = 2070326
      train_generator = trainGenerator.DataGenerator("train", myTrainFilePath, myTrainSize, batch_size=1200)
      
      myValidationFilePath = outDir+"FinalSampleCompressedAllImages_2017_validation_flattened_standardized_shuffled_v4.h5"
      myValidationSize = 257363
      validation_generator = trainGenerator.DataGenerator("validation", myValidationFilePath, myValidationSize, batch_size=1200)
      
      history = myModel.fit_generator(generator=train_generator, validation_data=validation_generator, steps_per_epoch=int(myTrainSize/BatchSize), epochs=50, validation_steps=int(myValidationSize/BatchSize), use_multiprocessing=True, workers=6)
      """
      
      """
      for myFrame in frameTypes:
         
         myTrainEvents = [] # format: for each of 5 frames new 6samplesxNEventsInSamplex31pixelsx31pixelsx1float
         myValidationEvents = [] # format: for each of 5 frames new 6samplesxNEventsInSamplesx31pixelsx31pixelsx1float
         if not "truthLabelsTrain" in globals():
            myTrainTruth = [] # format: 6samplesx6categoriesx1float
            myValidationTruth = [] # format: 6samplesx6categoriesx1float
            
         for mySample in sampleTypes:
            myTF = h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_test_flattened_standardized.h5","r")
            NTEvents = myTF[myFrame+"Frame_images"].shape[0]
            myTShape=(NTEvents,31,31,1)
            print("MyTrainEvents",len(myTrainEvents))
            myTrainEvents.append(numpy.array(myTF[myFrame+"Frame_images"][...,0]).resize(myTShape))
            print("MyTrainEvents",len(myTrainEvents))
            myTF.close()
            print("MyTrainEvents",len(myTrainEvents))
            if not "truthLabelsTrain" in globals():
               myTrainTruth.append(numpy.zeros((NTEvents,len(sampleTypes)))) # shape: NEventsInSamplex6Categoriesx1float filled w/ 0s
            print("MyTrainEvents",len(myTrainEvents))
            myVF = h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")
            NVEvents = myVF[myFrame+"Frame_images"].shape[0]
            myVShape=(NVEvents,31,31,1)
            myValidationEvents.append(numpy.array(myVF[myFrame+"Frame_images"][...,0]).resize(myVShape))
            myVF.close()
            if not "truthLabelsValidation" in globals():
               myValidationTruth.append(numpy.zeros((NVEvents,len(sampleTypes)))) # shape: NEventsInSamplex6Categoriesx1float filled w/ 0s
            print("MyValidationEvents",len(myValidationEvents))
               
         print("MyTrainEvents",len(myTrainEvents))
         print("MyValidationEvents",len(myValidationEvents))
         print("MyTrainTruth",len(myTrainTruth))
         print("MyValidationTruth",len(myValidationTruth))
         globals()["jet"+myFrame+"FrameTrain"] = numpy.concatenate((myTrainEvents[0],myTrainEvents[1],myTrainEvents[2],myTrainEvents[3],myTrainEvents[4],myTrainEvents[5]))
         globals()["jet"+myFrame+"FrameValidation"] = numpy.concatenate(myValidationEvents)
         if not "truthLabelsTrain" in globals():
            for i in range(len(sampleTypes)):
               myTrainTruth[i][:,i] = 1.
            globals()["truthLabelsTrain"] = numpy.concatenate(myTrainTruth)
         if not "truthLabelsValidation" in globals():
            for i in range(len(sampleTypes)):
               myValidationTruth[i][:,i] = 1.
            globals()["truthLabelsValidation"] = numpy.concatenate(myValidationTruth)
         
      """
      #myTrainBESEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["BES_vars"]) for mySample in sampleTypes]
      #myValidationBESEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["BES_vars"]) for mySample in sampleTypes]
      myTrainWEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["WFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationWEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["WFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainZEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["ZFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationZEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["ZFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainHiggsEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["HiggsFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationHiggsEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["HiggsFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainTopEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["TopFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationTopEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["TopFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainBottomEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["BottomFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationBottomEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["BottomFrame_images"][...,0]) for mySample in sampleTypes]
      #print("My Train events shape:",[myTrainBESEvents[i].shape for i in range(len(myTrainBESEvents))])
      #print("My Validation events shape:",[myValidationBESEvents[i].shape for i in range(len(myValidationBESEvents))])
      myTrainTruth = [numpy.zeros((len(myTrainWEvents[i]),len(sampleTypes))) for i in range(len(myTrainWEvents))] # shape: N,6 filled w/ 0s
      myValidationTruth = [numpy.zeros((len(myValidationWEvents[i]),len(sampleTypes))) for i in range(len(myValidationWEvents))]
      for i in range(len(sampleTypes)):
         myTrainTruth[i][:,i] = 1.
         myValidationTruth[i][:,i] = 1.
         
      print("My train truth shape:",[myTrainTruth[i].shape for i in range(len(myTrainTruth))])
      print("My validation truth shape:",[myValidationTruth[i].shape for i in range(len(myValidationTruth))]) 
      print("Labels are:",[[i,mySample] for i,mySample in enumerate(sampleTypes)])
      #globals()["jetBESvarsTrain"] = numpy.concatenate(myTrainBESEvents)
      globals()["jetWFrameTrain"] = numpy.concatenate(myTrainWEvents)
      globals()["jetZFrameTrain"] = numpy.concatenate(myTrainZEvents)
      globals()["jetHiggsFrameTrain"] = numpy.concatenate(myTrainHiggsEvents)
      globals()["jetTopFrameTrain"] = numpy.concatenate(myTrainTopEvents)
      globals()["jetBottomFrameTrain"] = numpy.concatenate(myTrainBottomEvents)
      globals()["truthLabelsTrain"] = numpy.concatenate(myTrainTruth)
      #globals()["jetBESvarsValidation"] = numpy.concatenate(myValidationBESEvents)
      globals()["jetWFrameValidation"] = numpy.concatenate(myValidationWEvents)
      globals()["jetZFrameValidation"] = numpy.concatenate(myValidationZEvents)
      globals()["jetHiggsFrameValidation"] = numpy.concatenate(myValidationHiggsEvents)
      globals()["jetTopFrameValidation"] = numpy.concatenate(myValidationTopEvents)
      globals()["jetBottomFrameValidation"] = numpy.concatenate(myValidationBottomEvents)
      globals()["truthLabelsValidation"] = numpy.concatenate(myValidationTruth)
      #print("Globals train shapes",globals()["jetBESvarsTrain"].shape, globals()["truthLabelsTrain"].shape, globals()["truthLabelsTrain"][0])
      #print("Globals validation shapes",globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
      
         
      print("Shuffle Train")
      rng_state = numpy.random.get_state()
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["truthLabelsTrain"])
      #numpy.random.set_state(rng_state)
      #numpy.random.shuffle(globals()["jetBESvarsTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBottomFrameTrain"])
      
      print("Shuffle Validation")
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["truthLabelsValidation"])
      #numpy.random.set_state(rng_state)
      #numpy.random.shuffle(globals()["jetBESvarsValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBottomFrameValidation"])

      print("Shape of truthLabelsTrain",globals()["truthLabelsTrain"].shape)
      print("Shape of truthLabelsValidation",globals()["truthLabelsValidation"].shape)
      for myFrame in frameTypes:
         print("Shape of "+myFrame+"FrameTrain", globals()["jet"+myFrame+"FrameTrain"].shape)
         print("Shape of "+myFrame+"FrameValidation", globals()["jet"+myFrame+"FrameValidation"].shape)

      
      
      #print("Globals train shapes",globals()["jetBESvarsTrain"].shape, globals()["truthLabelsTrain"].shape, globals()["truthLabelsTrain"][0])
      #print("Globals validation shapes",globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
      #quit()
      totalEventsTrain = len(globals()["truthLabelsTrain"])
      totalEventsValidation = len(globals()["truthLabelsValidation"])
      print("Batch Size: "+str(BatchSize)+", Epochs: 50"+", totalEventsTrain: "+str(totalEventsTrain)+", totalEventsValidation: "+str(totalEventsValidation))
      history = myModel.fit(
         [globals()["jetWFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetZFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetHiggsFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetTopFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetBottomFrameTrain"].reshape((totalEventsTrain,31,31,1))
         ],
         globals()["truthLabelsTrain"].reshape((totalEventsTrain,6)),
         batch_size=BatchSize, epochs=25, callbacks=[early_stopping, model_checkpoint],
         validation_data = [
            [globals()["jetWFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetZFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetHiggsFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetTopFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetBottomFrameValidation"].reshape((totalEventsValidation,31,31,1))
             ],
            globals()["truthLabelsValidation"].reshape((totalEventsValidation,6))
         ]
      )
      
      
   elif doBES and doImages:
      myTrainBESEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["BES_vars"]) for mySample in sampleTypes]
      myValidationBESEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["BES_vars"]) for mySample in sampleTypes]
      myTrainWEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["WFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationWEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["WFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainZEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["ZFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationZEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["ZFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainHiggsEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["HiggsFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationHiggsEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["HiggsFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainTopEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["TopFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationTopEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["TopFrame_images"][...,0]) for mySample in sampleTypes]
      myTrainBottomEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5","r")["BottomFrame_images"][...,0]) for mySample in sampleTypes]
      myValidationBottomEvents = [numpy.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["BottomFrame_images"][...,0]) for mySample in sampleTypes]
      #print("My Train events shape:",[myTrainBESEvents[i].shape for i in range(len(myTrainBESEvents))])
      #print("My Validation events shape:",[myValidationBESEvents[i].shape for i in range(len(myValidationBESEvents))])
      myTrainTruth = [numpy.zeros((len(myTrainWEvents[i]),len(sampleTypes))) for i in range(len(myTrainWEvents))] # shape: N,6 filled w/ 0s
      myValidationTruth = [numpy.zeros((len(myValidationWEvents[i]),len(sampleTypes))) for i in range(len(myValidationWEvents))]
      for i in range(len(sampleTypes)):
         myTrainTruth[i][:,i] = 1.
         myValidationTruth[i][:,i] = 1.
         
      print("My train truth shape:",[myTrainTruth[i].shape for i in range(len(myTrainTruth))])
      print("My validation truth shape:",[myValidationTruth[i].shape for i in range(len(myValidationTruth))]) 
      print("Labels are:",[[i,mySample] for i,mySample in enumerate(sampleTypes)])
      globals()["jetBESvarsTrain"] = numpy.concatenate(myTrainBESEvents)
      globals()["jetWFrameTrain"] = numpy.concatenate(myTrainWEvents)
      globals()["jetZFrameTrain"] = numpy.concatenate(myTrainZEvents)
      globals()["jetHiggsFrameTrain"] = numpy.concatenate(myTrainHiggsEvents)
      globals()["jetTopFrameTrain"] = numpy.concatenate(myTrainTopEvents)
      globals()["jetBottomFrameTrain"] = numpy.concatenate(myTrainBottomEvents)
      globals()["truthLabelsTrain"] = numpy.concatenate(myTrainTruth)
      globals()["jetBESvarsValidation"] = numpy.concatenate(myValidationBESEvents)
      globals()["jetWFrameValidation"] = numpy.concatenate(myValidationWEvents)
      globals()["jetZFrameValidation"] = numpy.concatenate(myValidationZEvents)
      globals()["jetHiggsFrameValidation"] = numpy.concatenate(myValidationHiggsEvents)
      globals()["jetTopFrameValidation"] = numpy.concatenate(myValidationTopEvents)
      globals()["jetBottomFrameValidation"] = numpy.concatenate(myValidationBottomEvents)
      globals()["truthLabelsValidation"] = numpy.concatenate(myValidationTruth)
      #print("Globals train shapes",globals()["jetBESvarsTrain"].shape, globals()["truthLabelsTrain"].shape, globals()["truthLabelsTrain"][0])
      #print("Globals validation shapes",globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
      
         
      print("Shuffle Train")
      rng_state = numpy.random.get_state()
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["truthLabelsTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameTrain"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBottomFrameTrain"])
      
      print("Shuffle Validation")
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["truthLabelsValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBESvarsValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetWFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetZFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetHiggsFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetTopFrameValidation"])
      numpy.random.set_state(rng_state)
      numpy.random.shuffle(globals()["jetBottomFrameValidation"])

      print("Shape of truthLabelsTrain",globals()["truthLabelsTrain"].shape)
      print("Shape of truthLabelsValidation",globals()["truthLabelsValidation"].shape)
      print("Shape of jetBESvarsTrain", globals()["jetBESvarsTrain"].shape)
      print("Shape of jetBESvarsValidation", globals()["jetBESvarsValidation"].shape)
      for myFrame in frameTypes:
         print("Shape of "+myFrame+"FrameTrain", globals()["jet"+myFrame+"FrameTrain"].shape)
         print("Shape of "+myFrame+"FrameValidation", globals()["jet"+myFrame+"FrameValidation"].shape)

      
      
      #print("Globals train shapes",globals()["jetBESvarsTrain"].shape, globals()["truthLabelsTrain"].shape, globals()["truthLabelsTrain"][0])
      #print("Globals validation shapes",globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
      #quit()
      totalEventsTrain = len(globals()["truthLabelsTrain"])
      totalEventsValidation = len(globals()["truthLabelsValidation"])
      print("Batch Size: "+str(BatchSize)+", Epochs: 50"+", totalEventsTrain: "+str(totalEventsTrain)+", totalEventsValidation: "+str(totalEventsValidation))
      history = myModel.fit(
         [globals()["jetWFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetZFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetHiggsFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetTopFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetBottomFrameTrain"].reshape((totalEventsTrain,31,31,1)),
          globals()["jetBESvarsTrain"].reshape((totalEventsTrain,142))
         ],
         globals()["truthLabelsTrain"].reshape((totalEventsTrain,6)),
         batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint],
         validation_data = [
            [globals()["jetWFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetZFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetHiggsFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetTopFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetBottomFrameValidation"].reshape((totalEventsValidation,31,31,1)),
             globals()["jetBESvarsValidation"].reshape((totalEventsValidation,142))
             ],
            globals()["truthLabelsValidation"].reshape((totalEventsValidation,6))
         ]
      )
      
   elif doBES and doImages and heavyComp:
      """
      print("Before Open File")
      myTrainFile = h5py.File(outDir+"FinalSampleCompressedAllImages_2017_train_flattened_standardized_shuffled_v4.h5","r")
      print("Before Open Second File")
      myValidationFile = h5py.File(outDir+"FinalSampleCompressedAllImages_2017_validation_flattened_standardized_shuffled_v4.h5","r")
      print("Opened Files")
      print("Shapes:",myTrainFile["jetWFrametrain"].shape, myTrainFile["jetZFrametrain"].shape, myTrainFile["jetHiggsFrametrain"].shape, myTrainFile["jetTopFrametrain"].shape, myTrainFile["jetBottomFrametrain"].shape, myTrainFile["jetBESvarstrain"].shape)
      print("fitting both bes and images")
      history = myModel.fit([myTrainFile["jetWFrametrain"][1000000:1500000], myTrainFile["jetZFrametrain"][1000000:1500000], myTrainFile["jetHiggsFrametrain"][1000000:1500000], myTrainFile["jetTopFrametrain"][1000000:1500000], myTrainFile["jetBottomFrametrain"][1000000:1500000], myTrainFile["jetBESvarstrain"][1000000:1500000]], myTrainFile["truthLabelstrain"][1000000:1500000], batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint], validation_data = [[myValidationFile["jetWFramevalidation"][0:10000], myValidationFile["jetZFramevalidation"][0:10000], myValidationFile["jetHiggsFramevalidation"][0:10000], myValidationFile["jetTopFramevalidation"][0:10000], myValidationFile["jetBottomFramevalidation"][0:10000], myValidationFile["jetBESvarsvalidation"][0:10000]], myValidationFile["truthLabelsvalidation"][0:10000]])
      """
      
      
      import trainGenerator
      
      myTrainFilePath = outDir+"FinalSampleCompressedAllImages_2017_train_flattened_standardized_shuffled_v4.h5"
      myTrainSize = 2070326 #1725 steps in an epoch
      train_generator = trainGenerator.DataGenerator("train", myTrainFilePath, myTrainSize, batch_size=BatchSize)
      
      #myValidationFilePath = outDir+"FinalSampleCompressedAllImages_2017_validation_flattened_standardized_shuffled_v4.h5"
      #myValidationSize = 257363
      #validation_generator = trainGenerator.DataGenerator("validation", myValidationFilePath, myValidationSize, batch_size=800)
      myValidationFile = h5py.File(outDir+"FinalSampleCompressedAllImages_2017_validation_flattened_standardized_shuffled_v4.h5","r")
      
      #history = myModel.fit_generator(generator=train_generator, validation_data=validation_generator, steps_per_epoch=int(myTrainSize/BatchSize), epochs=50, validation_steps=int(myValidationSize/BatchSize), use_multiprocessing=True, workers=6)
      history = myModel.fit_generator(generator=train_generator,
                                      steps_per_epoch=100,
                                      epochs=100,
                                      #validation_data=validation_generator#,
                                      validation_data=[[myValidationFile["jetWFramevalidation"][:],myValidationFile["jetZFramevalidation"][:],myValidationFile["jetHiggsFramevalidation"][:],myValidationFile["jetTopFramevalidation"][:],myValidationFile["jetBottomFramevalidation"][:],myValidationFile["jetBESvarsvalidation"][:]], myValidationFile["truthLabelsvalidation"][:]],
                                      #batch_size=BatchSize,
                                      #callbacks=[model_checkpoint]#,
                                      use_multiprocessing=True,
                                      workers=14
      )

   print("Trained the neural network!")

   # performance plots
   loss = [history.history['loss'], history.history['val_loss'] ]
   acc = [history.history['acc'], history.history['val_acc'] ]
   tools.plotPerformance(loss, acc, suffix)
   print("plotted BEST training Performance")

   return myModel

def ensemble(model_BES, model_Images, outDir, suffix, userPatience):

   ## PredictTrain1 should give an array of (NEvents, classification), BESvars
   ## PredictTrain2 should give an array of (NEvents, classification), Images
   ## Same for validation
   print("Making BES train predictions")
   predictTrainBES = model_BES.predict([globals()["jetBESvarsTrain"][:]])
   print("Making image train predictions")
   predictTrainImages = model_Images.predict([globals()["jetWFrameTrain"][:], globals()["jetZFrameTrain"][:], globals()["jetHiggsFrameTrain"][:], globals()["jetTopFrameTrain"][:]])
   print("Making validation BES predictions")
   predictValidationBES = model_BES.predict([globals()["jetBESvarsValidation"][:]])
   print("Making image validation predictions")
   predictValidationImages = model_Images.predict([globals()["jetWFrameValidation"][:], globals()["jetZFrameValidation"][:], globals()["jetHiggsFrameValidation"][:], globals()["jetTopFrameValidation"][:]])
   print("PredictTrainBES",predictTrainBES.shape, type(predictTrainBES))
   print("PredictTrainImages",predictTrainImages.shape, type(predictTrainImages))
   print("PredictValidationBES",predictValidationBES.shape, type(predictValidationBES))
   print("PredictTrainImages",predictValidationImages.shape, type(predictValidationImages))

   ## Need to make new network combining output of other networks here
   EnsembleShapeHolder = 12 #Six category weights for images and six for BES
   ensembleInputs = Input( shape=(EnsembleShapeHolder,) )
   ensembleModel = Model(inputs = ensembleInputs, outputs = ensembleInputs)
      
   ensemble = ensembleModel.output
   '''
   ensembleLayer = Dense(512, kernel_initializer="glorot_normal", activation="relu" )(ensemble)
   ensembleLayer = Dropout(0.20)(ensembleLayer)# try 0.35
   ensembleLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   #Another dropout of 0.35
   ensembleLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   ensembleLayer = Dense(256, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   '''
   ensembleLayer = Dense(144, kernel_initializer="glorot_normal", activation="relu" )(ensemble)
   ensembleLayer = Dropout(0.22)(ensembleLayer)#was 0.10
   ensembleLayer = Dense(72, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   ensembleLayer = Dropout(0.22)(ensembleLayer)#was 0.20
   ensembleLayer = Dense(36, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   ensembleLayer = Dropout(0.22)(ensembleLayer)#was 0.20
   ensembleLayer = Dense(18, kernel_initializer="glorot_normal", activation="relu" )(ensembleLayer)
   ensembleLayer = Dropout(0.22)(ensembleLayer)#was 0.20
   outputEnsemble = Dense(6, kernel_initializer="glorot_normal", activation="softmax")(ensembleLayer)
      
   model_Ensemble = Model(inputs = [ensembleModel.input], outputs = outputEnsemble)
   model_Ensemble.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   print(model_Ensemble.summary() )
   # early stopping
   early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=userPatience, verbose=0, mode='auto', restore_best_weights=True)
   model_checkpointEnsemble = ModelCheckpoint(outDir+'BEST_model'+suffix+'.h5', monitor='val_loss', 
                                              verbose=0, save_best_only=True, 
                                              save_weights_only=False, mode='auto', 
                                              period=1)
   
   concatTrain = numpy.concatenate((predictTrainBES[:], predictTrainImages[:]),axis=1)
   concatValidation = numpy.concatenate((predictValidationBES[:], predictValidationImages[:]),axis=1)
   print("concatTrain", concatTrain.shape, type(concatTrain))
   print("concatValidation", concatValidation.shape, type(concatValidation))
   print("truthLabelsTrain", globals()["truthLabelsTrain"].shape, type(globals()["truthLabelsTrain"]))
   print("truthLabelsValidation", globals()["truthLabelsValidation"].shape, type(globals()["truthLabelsValidation"]))
   historyEnsemble = model_Ensemble.fit([concatTrain], globals()["truthLabelsTrain"][:], batch_size=BatchSize, epochs=200, callbacks=[early_stopping, model_checkpointEnsemble], validation_data = [[concatValidation], globals()["truthLabelsValidation"][:]])
   lossEnsemble = [historyEnsemble.history['loss'], historyEnsemble.history['val_loss'] ]
   accEnsemble = [historyEnsemble.history['acc'], historyEnsemble.history['val_acc'] ]
   tools.plotPerformance(lossEnsemble, accEnsemble, suffix)

   print("Trained Ensembler")
   return model_Ensemble


# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="~/nobackup/h5samples/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="~/nobackup/models/")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="")
    parser.add_argument('-y','--year',
                        dest='year',
                        default="2017")
    parser.add_argument('-p','--patience',
                        dest='patience',
                        default="20")
    parser.add_argument('-b','--doBES', dest='doBES', default=False, action='store_true')
    parser.add_argument('-i','--doImages', dest='doImages', default=False, action='store_true')
    parser.add_argument('-e','--doEnsemble', dest='doEnsemble', default=False, action='store_true')
    parser.add_argument('-r','--redoTraining', dest='redoTraining', default=False, action='store_true')
    parser.add_argument('-ob','--oldBEST', dest='oldBEST', default=False, action='store_true')
    args = parser.parse_args()
   
    if args.doBES: doBES = True
    if args.doImages: doImages = True
    if args.doEnsemble: doEnsemble = True
    if doEnsemble:
        doBES = True
        doImages = True
    mySuffix = args.suffix+args.year
    if args.oldBEST:
        doBES = True
        doImages = False
        doEnsemble = False
        mySuffix = mySuffix + "_oldBEST"
    """
    else:
        if doBES and not doImages:
            mySuffix = mySuffix + "_BESonly"
        elif not doBES and doImages:
            mySuffix = mySuffix + "_Imagesonly"
        else:
            mySuffix = mySuffix + "_BothBESandImages"
    """ 
    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isdir(args.outDir):
        print(args.outDir, "does not exist")
        quit()

    if args.redoTraining:
        print("Redo all training")
        #loadData(args.h5Dir, args.year, ["Train","Validation"])
        # If ensemble enabled, BES and Images trained separately, then ensembler is trained based on their output
        if doEnsemble:
            BES_model = train(doBES, False, args.outDir, mySuffix+"_BES", float(args.patience))
            Images_model = train(False, doImages, args.outDir, mySuffix+"_Images", float(args.patience))
            BEST_model = ensemble(BES_model, Images_model, args.outDir, mySuffix+"_Ensemble", float(args.patience))
        # If ensemble NOT enabled and both BES and images enabled, the result is a BES+Images network.
        else:
            if args.oldBEST:
                BEST_model = train(doBES, doImages, args.h5Dir, args.outDir, mySuffix+"_oldBEST", float(args.patience))
            elif doBES and not doImages:
                BEST_model = train(doBES, doImages, args.h5Dir, args.outDir, mySuffix+"_BES", float(args.patience))
            elif not doBES and doImages:
                BEST_model = train(doBES, doImages, args.h5Dir, args.outDir, mySuffix+"_Images", float(args.patience))
            else:
                BEST_model = train(doBES, doImages, args.h5Dir, args.outDir, mySuffix+"_Both", float(args.patience)) 
    else:
        print("Finding models available, training what is missing")
        if doEnsemble:
            if not os.path.isfile(args.outDir+"BEST_model"+mySuffix+"_BES.h5"):
                print("Train BESvars model")
                if not "jetBESvarsTrain" in globals().keys():
                    loadData(args.h5Dir, args.year, ["Train","Validation"])
                BES_model = train(doBES, False, args.outDir, mySuffix+"_BES", float(args.patience))
            else:
                print("Loading BES model")
                BES_model = load_model(args.outDir+"BEST_model"+mySuffix+"_BES.h5")
            if not os.path.isfile(args.outDir+"BEST_model"+mySuffix+"_Images.h5"):
                print("Train Image model")
                if not "jetWFrameTrain" in globals().keys():
                    loadData(args.h5Dir, args.year, ["Train","Validation"])
                Images_model = train(False, doImages, args.outDir, mySuffix+"_Images", float(args.patience))
            else:
                print("Loading Images model")
                Images_model = load_model(args.outDir+"BEST_model"+mySuffix+"_Images.h5")
            if not os.path.isfile(args.outDir+"BEST_model"+mySuffix+"_Ensemble.h5"):
                print("Train Ensemble model")
                if not "jetBESvarsTrain" in globals().keys():
                    loadData(args.h5Dir, args.year, ["Train","Validation"])
                BEST_model = ensemble(BES_model, Images_model, args.outDir, mySuffix+"_Ensemble", float(args.patience))
            else:
                print("Loading Ensemble model")
                BEST_model = load_model(args.outDir+"BEST_model"+mySuffix+"_Ensemble.h5")
        else:
            BEST_model = None
            if args.oldBEST:
                #BEST_model = train(doBES, doImages, args.h5Dir, args.outDir, mySuffix+"_oldBEST", float(args.patience))
                if not os.path.isfile(args.outDir+"BEST_model"+mySuffix+"_oldBEST.h5"):
                    print(args.outDir+"BEST_model"+mySuffix+"_oldBEST.h5"+" model not found, training a new one")
                    if not "jetBESvarsTrain" in globals().keys():
                        loadData(args.h5Dir, args.year, ["Train","Validation"])
            elif doBES and not doImages:
                if not os.path.isfile(args.outDir+"BEST_model"+mySuffix+".h5"):
                    print("BESonly model not found, training a new one")
                    if not "jetBESvarsTrain" in globals().keys():
                        loadData(args.h5Dir, args.year, ["Train","Validation"])
                else:
                    print("Loading BES-only model")
                    BEST_model = load_model(args.outDir+"BEST_model"+mySuffix+".h5")
            elif not doBES and doImages:
                if not os.path.isfile(args.outDir+"BEST_model"+mySuffix+".h5"):
                    print("Imagesonly model not found, training a new one")
                    if not "jetWFrameTrain" in globals().keys():
                        loadData(args.h5Dir, args.year, ["Train","Validation"])
                else:
                    print("Loading Images-only model")
                    BEST_model = load_model(args.outDir+"BEST_model"+mySuffix+".h5")
            else:
                if not os.path.isfile(args.outDir+"BEST_model"+mySuffix+".h5"):
                    print("Combined (both BESvars and Images, but not ensembled) model not found, training a new one")
                    if not "jetBESvarsTrain" in globals().keys():
                        loadData(args.h5Dir, args.year, ["Train","Validation"])
                else:
                    print("Loading combined BES-images (no ensemble) model")
                    BEST_model = load_model(args.outDir+"BEST_model"+mySuffix+".h5")
            if BEST_model == None:
                if args.oldBEST:
                    mySuffix = mySuffix+"_oldBEST"
                BEST_model = train(doBES, doImages, args.h5Dir, args.outDir, mySuffix, float(args.patience))

    for mySet in ["Train","Validation"]:
        if doImages:
            for myFrame in frameTypes:
                if "jet"+myFrame+"Frame"+mySet in globals().keys():
                    del globals()["jet"+myFrame+"Frame"+mySet]
        if doBES:
            if "jetBESvars"+mySet in globals().keys():
                del globals()["jetBESvars"+mySet]

   #if doBES:
   #   makeCM(BES_model, doBES, doImages, doEnsemble, mySuffix+"_BES")
   #if doImages:
   #   makeCM(Images_model, doBES, doImages, doEnsemble, mySuffix+"_Images")

   
   
    makeCM(BEST_model, args.h5Dir, args.outDir, args.year, doBES, doImages, doEnsemble, mySuffix)
