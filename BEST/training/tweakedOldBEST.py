#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# tweakedOldBEST.py ///////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Reyer Band, Johan S. Bonilla, Brendan Regnary, Mark Samuel Abbott ////
# This program trains BEST with flattened inputs //////////////////////////////////
# This uses a tweaked version of the oldBEST NN architecture //////////////////////
#==================================================================================

################################## NOTES TO SELF ##################################
# Check for conistency, add comments.

import time
startTime = time.time() # Tracks how long script takes


# modules
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import tensorflow as tf
import random
import numpy.random
import math
# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" # must set backend before importing keras
# set up gpu environment
from keras import backend as k
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
k.tensorflow_backend.set_session(tf.Session(config=config))

from keras.models import load_model

# user modules
import tools.functions as tools
from plotConfusionMatrix import makeCM

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))
sampleTypes = ["Signal","QCD","TTBar"]
#sampleTypes = ["Zt","Ht","Wb","QCD","TTBar"]
frameTypes = [""]
maxEvents = {"train":None,      "validation":None,   "test":None}
BatchSize = 1200
# TrnValTstEvents = [ None, None, None ]
# TrnValTstEvents = [ 2000000, 200000, 200000 ]
TrnValTstEvents = [ 150000, 150000, 150000 ]
# TrnValTstEvents = [ 250000, 25000, 25000 ]
# TrnValTstEvents = [ 125000, 12500, 12500 ]
# TrnValTstEvents = [ 50000, 10000, 10000 ]

def train(h5Dir, modelFile, plotDir, suffix, userPatience, maskPath, TrnValTstEvents):
    #==================================================================================
    # Train the Neural Network ////////////////////////////////////////////////////////
    #==================================================================================

    # Keep updating comments, sections, explanations
    # How can this be sped up?
    #kedOldBEST.py    Batches? Don't load more data than needed
    # How can this be simplified?
    # How can this be split up into easy to read functions?
    # Keep redo trtweakedOldBEST.pyaining?
    # Restructure how functions are called--consolodate. Will need to understand final file structure (how many training scripts?)
    #   Currently a mess. functions.py and plotConfustionMatrix.py should be combined. Unneeded things should be deleted.


    # Create the BES framework
    # Train the neural network
    # maskDir = "/uscms/home/msabbott/nobackup/abbott/CMSSW_10_6_27/src/BEST/formatConverter/masks/"
    # maskName = "oldBESTMask.txt"
    # maskFile = open(maskDir + maskName, "r")

    maskFile = open(maskPath, "r")
    maskIndex = []
    for line in maskFile:
        maskIndex.append(line.split(':')[0])
    maskFile.close()
    print(maskPath + " chosen; mask size " + str(len(maskIndex)))
    myMask = [True if str(i) in maskIndex else False for i in range(82)]


    besInputs = Input( shape=(len(maskIndex), ) )        
    besModel  = Model( inputs = besInputs, outputs = besInputs )
    print(besModel.output)   

    # Add BES variables to the network
    combined = besModel.output

    # # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    combLayer   = Dense(82, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    # combLayer   = Dropout(0.15)(combLayer) # try 0.35
    combLayer   = Dense(82, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # combLayer   = Dropout(0.35)(combLayer) # try 0.35
    combLayer   = Dense(82, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # combLayer   = Dropout(0.25)(combLayer) # try 0.35    
    outputModel = Dense(  3, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    # combLayer   = Dense(10, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    # combLayer   = Dense(10, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # combLayer   = Dense(10, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # outputModel = Dense( 6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    # combLayer   = Dense(1000, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    # combLayer   = Dense(1000, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # combLayer   = Dense(1000, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # outputModel = Dense(   6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # Compile the model
    myModel = Model(inputs = [besModel.input], outputs = outputModel)
    myModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(myModel.summary() )

    # Early stopping
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=userPatience, verbose=0, mode='auto')#, restore_best_weights=True,)
    #min delta started as 0.01

    # Model checkpoint callback
    # This saves the model architecture + parameters into dense_model.h5
    # model_checkpoint = ModelCheckpoint( modelFile, monitor='val_loss', 
    #                                     verbose=0, save_best_only=True, 
    #                                     save_weights_only=False, mode='auto', 
    #                                     period=1)
    model_checkpoint = ModelCheckpoint( modelFile, monitor='val_loss', 
                                        verbose=1, save_best_only=False,
                                        save_weights_only=False,
                                        period=1, mode='auto')

    # Train the neural network


    myTrainEvents      = [np.array(h5py.File(h5Dir+mySample+"Sample_2018_BESTinputs_train_flattened_standardized_maxAbs.h5",     "r")["BES_vars"])[:150000,myMask] for mySample in sampleTypes]
    #myValidationEvents = myTrainEvents
    myValidationEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2018_BESTinputs_validation_flattened_standardized_maxAbs.h5","r")["BES_vars"])[:15000,myMask] for mySample in sampleTypes]

    # myTrainEvents      = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened_standardized.h5",     "r")["BES_vars"])[:,:] for mySample in sampleTypes]
    # myValidationEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_validation_flattened_standardized.h5","r")["BES_vars"])[:,:] for mySample in sampleTypes]

    print("My Train events shape:",      [myTrainEvents[i].shape      for i in range(len(myTrainEvents))])
    print("My Validation events shape:", [myValidationEvents[i].shape for i in range(len(myValidationEvents))])

    myTrainTruth      = [np.zeros( (len(myTrainEvents[i]),      len(sampleTypes)) )      for i in range(len(myTrainEvents))] # shape: N,6 filled w/ 0s
    myValidationTruth = [np.zeros( (len(myValidationEvents[i]), len(sampleTypes)) )      for i in range(len(myValidationEvents))]
    for i in range(len(sampleTypes)):
        myTrainTruth[i][:,i] = 1.
        myValidationTruth[i][:,i] = 1.

    print("My train truth shape:",      [ myTrainTruth[i].shape      for i in range(len(myTrainTruth))])
    print("My validation truth shape:", [ myValidationTruth[i].shape for i in range(len(myValidationTruth))]) 

    print("Labels are:", [ [i,mySample] for i,mySample in enumerate(sampleTypes)])
    
    globals()["jetBESvarsTrain"]       = np.concatenate(myTrainEvents)
    globals()["truthLabelsTrain"]      = np.concatenate(myTrainTruth)
    globals()["jetBESvarsValidation"]  = np.concatenate(myValidationEvents)
    globals()["truthLabelsValidation"] = np.concatenate(myValidationTruth)
    
    print("Globals train shapes",      globals()["jetBESvarsTrain"].shape,      globals()["truthLabelsTrain"].shape,      globals()["truthLabelsTrain"][0])
    print("Globals validation shapes", globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
         
    print("Shuffle Train")
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["truthLabelsTrain"])
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["jetBESvarsTrain"])
    
    print("Shuffle Validation")
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["truthLabelsValidation"])
    np.random.set_state(rng_state)
    np.random.shuffle(globals()["jetBESvarsValidation"])
      
    print("Globals train shapes",      globals()["jetBESvarsTrain"].shape,      globals()["truthLabelsTrain"].shape,      globals()["truthLabelsTrain"][0])
    print("Globals validation shapes", globals()["jetBESvarsValidation"].shape, globals()["truthLabelsValidation"].shape, globals()["truthLabelsValidation"][0])
    # quit()
    print("Batch Size: " + str(BatchSize) + ", Epochs: 50")

    # history = myModel.fit(  [globals()["jetBESvarsTrain"][0:500000] ], globals()["truthLabelsTrain"][0:500000], 
   #                         batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint], 
    #                         validation_data = [[globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]])
    history = myModel.fit(  [globals()["jetBESvarsTrain"][:] ], globals()["truthLabelsTrain"][:],
                        # batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint],
                         batch_size=BatchSize, epochs=200, callbacks=[ model_checkpoint],
                         validation_data = [[globals()["jetBESvarsValidation"][:]], globals()["truthLabelsValidation"][:]], 
			shuffle=True,steps_per_epoch=None)

    trainMaxEvents      = TrnValTstEvents[0]
    validationMaxEvents = TrnValTstEvents[1]
   # history = myModel.fit(  [globals()["jetBESvarsTrain"][:trainMaxEvents] ], globals()["truthLabelsTrain"][:trainMaxEvents],
                        #batch_size=BatchSize, epochs=50, callbacks=[early_stopping, model_checkpoint],
                        #validation_data = [[globals()["jetBESvarsValidation"][:validationMaxEvents]], globals()["truthLabelsValidation"][:validationMaxEvents]])



    print("Trained the neural network!")

    # performance plots
    loss = [history.history['loss'], history.history['val_loss'] ]
    acc  = [history.history['acc'],  history.history['val_acc']  ]
    tools.plotPerformance(loss, acc, suffix, plotDir)
    print("Plotted BEST training Performance")

    return myModel


# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="")
    parser.add_argument('-m','--maskPath',
                        dest='maskPath',
                        default="")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="")
    parser.add_argument('-y','--year',
                        dest='year',
                        default="2018")
    parser.add_argument('-p','--patience',
                        dest='patience',
                        default="20")
    
    parser.add_argument('-t','--train', dest='train',
                        action='store_false',
                        help="Boolean, use flag to load an already trained model and plot it. [default: True]")    
    parser.add_argument('-sc','--scale', dest='scale',
                        default="standardized", # default="newBEST_Basic"
                        help="String used by MakeStandardInputs.py to name scaled data files [default: standardized]")
    parser.add_argument('-mt','--modelType', dest='modelType',
                        default="tweakedOldBEST",
                        help="Name of directory within models/ and plots/ [default: nnBEST]")
    parser.add_argument('-tol','--tolerance', dest='tolerance',      
                        # default="0.01",
                        default="0.001",
                        help="Improvement tolerance for EarlyStopping; smaller tolerance means longer training [default: 0.01]")
    parser.add_argument('-n','--nodes', dest='nodes',
                        default="140",                     
                        help="Number of nodes per hidden layer [default: 140]")
    parser.add_argument('-r','--replace', dest='replace',
                        action='store_true',
                        help="Boolean, use flag to overwrite current model and plots [default: False]")
    

    # parser.add_argument('-ev','--events',
    #                     dest='events',
    #                     default="")
    parser.add_argument('-rt','--redoTraining', dest='redoTraining', default=False, action='store_true')
    args = parser.parse_args()

    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isdir(args.outDir):
        print(args.outDir, "does not exist")
        quit()


    modelType = "tweakedOldBEST"
    maskName  = args.maskPath[1 + args.maskPath.rfind("/"):] # Strip everything after the final '/', giving just the name of the mask
    mySuffix = args.suffix + args.year + "_" + maskName[:-4] + "_" + modelType
    plotDir  = "plots/" + modelType + "/valloss/" + mySuffix
    modelDir = args.outDir + "/" + modelType + "/" + mySuffix
    modelFile = modelDir + "/BEST_model_" + mySuffix + ".h5"
    maskSave  = modelDir + "/" + maskName

    # mySuffix  = args.suffix + args.year + "_oldBEST"
    # modelFile = args.outDir + "BEST_model_" + mySuffix+ ".h5"
    # maskPath = "/uscms/home/msabbott/nobackup/abbott/CMSSW_10_6_27/src/BEST/formatConverter/masks/oldBESTMask.txt"

    if args.redoTraining:
        print("Redo all training")
        BEST_model = train(args.h5Dir, modelFile, plotDir, mySuffix, float(args.patience), args.maskPath, TrnValTstEvents)
    else:
        print("Begin training new model...")
        if not os.path.isdir(modelDir):
            print("Creating directory for model and mask: " + modelDir )
            os.makedirs(modelDir)
        elif os.path.isfile(modelFile): 
            print("Replacing " + modelFile)
            os.remove(modelFile)
            print("Replacing " + maskSave)
            os.remove(maskSave)
        print("Copying mask into model directory...")
        os.system('cp ' + args.maskPath + ' ' + maskSave)
        BEST_model = train(args.h5Dir, modelFile, plotDir, mySuffix, float(args.patience), args.maskPath, TrnValTstEvents)

    for mySet in ["Train","Validation"]:
        if "jetBESvars"+mySet in globals().keys():
            del globals()["jetBESvars"+mySet]

    testMaxEvents = TrnValTstEvents[2]
    print("deleting model") 
    del BEST_model
    print("Ready to load model")
    BEST_model = load_model(modelFile)
    print("Made it to the point where CM is being made")
    makeCM(load_model(modelFile), args.h5Dir, plotDir, mySuffix, args.maskPath, testMaxEvents, modelType)
    print("survived past the making of CM")
    # Check how long the script took to run
    timelog = open("logs/timelog_" + modelType, "a") 
    timeTaken = divmod(time.time() - startTime, 60.)
    timeMessage = "Running " + mySuffix + ", script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
    print(timeMessage)
    

    #new stuff
    strings = tools.dirStrings(args)
    mask = tools.loadMask(args.maskPath)
    #if args.train: # Run with -t to only plot the performance of the model
        # Load h5 Data, set up truth arrays
    #    dataDict = tools.loadH5Data(args, mask, sampleTypes, ["train", "validation"], maxEvents) 

        # Shuffle arrays
        #tools.shuffleArray(dataDict)

        # Train using nnBEST
        #trainNNBEST(args, strings, mask, dataDict)

    # Load test data, evaluate model performance
    dataDict = tools.loadH5Data(args, mask, sampleTypes, ["test"], maxEvents)
    
    tools.plotAll(strings, dataDict, args.modelType, mask)
    timelog.write(timeMessage)
    timelog.close

