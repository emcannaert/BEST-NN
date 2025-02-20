#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# oldBEST.py //////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Samantha Abbott, Reyer Band, Johan S. Bonilla, Brendan Regnary  //////
# This program trains BEST with flattened inputs //////////////////////////////////
# This uses the original oldBEST NN architecture //////////////////////////////////
#==================================================================================

# user modules
import training.tools.functions_test as tools
startTime = tools.logTime() # Tracks how long script takes

# modules
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import tensorflow as tf
from sklearn.externals.joblib import dump

# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" # must set backend before importing keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
k.tensorflow_backend.set_session(tf.Session(config=config))

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]

setTypes = ["train", "validation", "test"]

# maxEvents is the max number of events to pull from EACH of the 6 sample files
# "None" means use all of events in each file 
maxEvents = {"train":None,      "validation":None,   "test":None}
# maxEvents = {"train":2000000, "validation":200000, "test":None}
# maxEvents = {"train":250000,  "validation":25000,  "test":None}
# maxEvents = {"train":50000,   "validation":10000,  "test":None}

def trainOldBEST(strings, pat, mask, dataDict):
    print("Begin training oldBEST")

    # Unpack strings for readability
    modelFile = strings["modelFile"] 
    historyFile = strings["historyFile"] 
    # plotDir = strings["plotDir"] # not used in this function 
    # suffix = strings["suffix"] # not used in this function 

    #==================================================================================
    # Train the Neural Network ////////////////////////////////////////////////////////
    #==================================================================================
    BatchSize = 1200

    # Create the BES framework

    # Input variables, shape = number of 'True' entries in mask
    besInputs = Input( shape=(np.array(mask).sum(), ) )        
    besModel  = Model( inputs = besInputs, outputs = besInputs )
    print(besModel.output)   

    # Add BES variables to the network
    combined = besModel.output

    # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    combLayer   = Dense(80, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    combLayer   = Dense(80, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    combLayer   = Dense(80, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    outputModel = Dense( 6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # Compile the model
    myModel = Model(inputs = [besModel.input], outputs = outputModel)
    myModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(myModel.summary() )

    # Train the neural network
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=pat, verbose=0, mode='auto')#, restore_best_weights=True,)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=pat, verbose=0, mode='auto')#, restore_best_weights=True,)

    # Model checkpoint callback
    # This saves the model architecture + parameters into a .pb file
    model_checkpoint = ModelCheckpoint( modelFile, monitor='val_loss', 
                                        verbose=1, save_best_only=True,
                                        save_weights_only=False,
                                        period=1, mode='auto')

    history = myModel.fit( [dataDict["trainEvents"]], dataDict["trainTruth"], 
                            validation_data = [[dataDict["validationEvents"]], dataDict["validationTruth"]],
                            batch_size=BatchSize, epochs=200, 
                            callbacks=[early_stopping, model_checkpoint],
                            shuffle = True, steps_per_epoch=None
                         ) 

    print("Trained the neural network!")
    del dataDict

    dump(history.history, historyFile)

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to train neural network.')
    parser.add_argument('-hd','--h5Dir', dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/",
                        help="Input File Dir [default: /uscms/home/bonillaj/nobackup/h5samples_ULv1/]")
    parser.add_argument('-o','--outDir', dest='outDir',
                        default="models/",
                        help="Output Dir where models are saved [default: models/]")
    parser.add_argument('-mp','--maskPath', dest='maskPath',
                        default = "../formatConverter/masks/oldBESTMask.txt",
                        help="Path to mask file [default: ../formatConverter/masks/oldBESTMask.txt]")
    parser.add_argument('-sf','--suffix', dest='suffix',
                        default="",
                        help="Suffix, used to uniquely identify model [default: '']")
    parser.add_argument('-sc','--scale', dest='scale',
                        default="standardized",
                        help="String used by MakeStandardInputs.py to name scaled data files [default: standardized]")
    parser.add_argument('-mt','--modelType', dest='modelType',
                        default="oldBEST",
                        help="Name of directory within models/ and plots/ [default: oldBEST]")
    parser.add_argument('-y','--year', dest='year',
                        default="2017",
                        help="Year of data taking to use [default: 2017]")
    parser.add_argument('-p','--patience', dest='patience',
                        default="20",
                        help="Number of Epochs to wait for improvement before EarlyStopping [default: 20]")
    parser.add_argument('-r','--replace', dest='replace',
                        action='store_true',
                        help="Boolean, use flag to overwrite current model and plots [default: False]")
    parser.add_argument('-t','--train', dest='train',
                        action='store_false',
                        help="Boolean, use flag to load an already trained model and plot it. [default: True]")                         
    args = parser.parse_args()

    # Generate appropriate helper strings, check dirs
    strings = tools.dirStrings(args)

    # Load Mask
    mask, _ = tools.loadMask(args.maskPath)

    # To skip training and plot the performance of an already trained model, use -t flag.
    if args.train: # Train the model 
        # Load h5 Data, set up truth arrays
        dataDict = tools.loadH5Data(args, mask, sampleTypes, ["train", "validation"], maxEvents) 

        # Shuffle arrays
        tools.shuffleArray(dataDict)

        # Train using oldBEST
        trainOldBEST(strings, float(args.patience), mask, dataDict)

    # Load test data for evaluating model performance
    dataDict = tools.loadH5Data(args, mask, sampleTypes, ["test"], maxEvents)

    # Make all performance plots
    tools.plotAll(strings, dataDict, args.modelType, mask)
    
    tools.logTime(startTime)