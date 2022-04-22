#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# oldBEST.py //////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Reyer Band, Johan S. Bonilla, Brendan Regnary, Sam Abbott ////////////
# This program trains BEST with flattened inputs //////////////////////////////////
# This uses the original oldBEST NN architecture //////////////////////////////////
#==================================================================================

################################## NOTES TO SELF ##################################
# Check for conistency, add comments.

import time
startTime = time.time() # Tracks how long script takes


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

# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
k.tensorflow_backend.set_session(tf.Session(config=config))

# user modules
# import tools.functions as tools
import tools.functions as tools
from plotConfusionMatrix import plotAll

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
frameTypes = ["W","Z","Higgs","Top","Bottom"]

#try different batchsizes
BatchSize = 1200
# TrnValEvents = [ None, None]
# TrnValEvents = [ 2000000, 200000]
TrnValEvents = [ 500000, 50000]
# TrnValEvents = [ 250000, 25000]
# TrnValEvents = [ 125000, 12500]
# TrnValEvents = [ 50000, 10000]

trainMaxEvents      = TrnValEvents[0]
validationMaxEvents = TrnValEvents[1]

setTypes = ["validation", "train"]

def train(modelFile, plotDir, suffix, userPatience, mask, scalePath, dataDict, nodes):
    #==================================================================================
    # Train the Neural Network ////////////////////////////////////////////////////////
    #==================================================================================

    # Create the BES framework
    # Train the neural network

    # Input variables, shape = number of 'True' entries in mask
    besInputs = Input( shape=(np.array(mask).sum(), ) )        
    besModel  = Model( inputs = besInputs, outputs = besInputs )
    print(besModel.output)   

    # Add BES variables to the network
    combined = besModel.output

    # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    # combLayer   = Dense(40, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    # combLayer   = Dense(40, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # combLayer   = Dense(40, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # outputModel = Dense( 6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    # combLayer   = Dense(111, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    # combLayer   = Dense(111, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # combLayer   = Dense(111, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # outputModel = Dense(  6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    outputModel = Dense( 6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # Compile the model
    myModel = Model(inputs = [besModel.input], outputs = outputModel)
    myModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(myModel.summary() )

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=userPatience, verbose=0, mode='auto')#, restore_best_weights=True,)

    # Model checkpoint callback
    # This saves the model architecture + parameters into dense_model.h5
    # model_checkpoint = ModelCheckpoint( modelFile, monitor='val_loss', 
    #                                     verbose=0, save_best_only=True, 
    #                                     save_weights_only=False, mode='auto', 
    #                                     period=1)
    model_checkpoint = ModelCheckpoint( modelFile, monitor='loss', 
                                        verbose=1, save_best_only=False,
                                        save_weights_only=False,
                                        period=1, mode='auto')


    # Shuffle arrays
    rng_state = np.random.get_state()
    tools.shuffleArray(dataDict, rng_state)
    
    # Load scaler model  
    scaler = tools.loadScalerModel(scalePath, mask)

    print("Scaling validation...")
    scaledValidEvents = scaler.transform(dataDict["validationEvents"][:validationMaxEvents])
    print("Scaling train...")
    scaledTrainEvents = scaler.transform(dataDict["trainEvents"][:trainMaxEvents])
    del scaler

    truthValid = dataDict["validationTruth"][:validationMaxEvents]
    truthTrain = dataDict["trainTruth"][:trainMaxEvents]

    print("Input validation shapes", scaledValidEvents.shape, truthValid.shape, truthValid[0] )
    print("Input train shapes",      scaledTrainEvents.shape, truthTrain.shape, truthTrain[0] )
    print("Batch Size: " + str(BatchSize) + ", Epochs: 50")

    history = myModel.fit( [scaledTrainEvents], truthTrain, batch_size=BatchSize, 
                            epochs=50, callbacks=[early_stopping, model_checkpoint],
                            validation_data = [[scaledValidEvents], truthValid] )

    print("Trained the neural network!")
    del scaledTrainEvents
    del scaledValidEvents
    del truthTrain
    del truthValid

    # Record max accuracy
    accIndex = np.argmax(history.history['acc'])
    accLog = open("Logs/" + modelType + "_accuracyLog.txt", "a") 
    # accLog.write("Ran " + suffix + ", max acc = " + str(np.max(history.history['acc'])) + '\n')
    accLog.write("Ran " + suffix + ": Max acc = " + str(history.history['acc'][accIndex])[:6] + ",  val_acc = " + str(history.history['val_acc'][accIndex])[:6] + '\n')
    accLog.close

    # Accuracy and Loss plots
    loss = [history.history['loss'], history.history['val_loss'] ]
    acc  = [history.history['acc'],  history.history['val_acc']  ]
    tools.plotPerformance(loss, acc, suffix, plotDir)
    print("Plotted BEST training Performance")
    # return myModel
    del history
    del myModel
    del outputModel
    del combLayer
    del combined
    del besModel
    del besInputs


# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        # default="~/nobackup/models/")
                        default="models")
    parser.add_argument('-m','--maskPath',
                        dest='maskPath',
                        default = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_noDeepAK8noNJets_trimIso.txt")
                        # default = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask.txt")
                        # default = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask_noEtaCharge.txt")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="")
    parser.add_argument('-y','--year',
                        dest='year',
                        default="2017")
    parser.add_argument('-p','--patience',
                        dest='patience',
                        default="20")
    parser.add_argument('-r','--redoTraining', dest='redoTraining', default=False, action='store_true')
    args = parser.parse_args()

    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isdir(args.outDir):
        print(args.outDir, "does not exist")
        quit()

    # suffixes = {"20":[20,"newBESTMask_noDeepAK8noNJets_trimIso.txt"], "80":[80,"newBESTMask_noDeepAK8noNJets_trimIso.txt"] }#, 
                # "Choked":[10,"oldBESTMask.txt"], "Big":[1000,"oldBESTMask.txt"]}
    # for suffix, params in suffixes.items():
    # choked and big for old and new best all scales

    # color outputs thoooooo?????
    scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    maskDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/"
    maskList = ["oldBESTMask.txt", "newBESTMask_noDeepAK8noNJets_trimIso.txt"]
    nodeList = [30,50,60,70,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,
                310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500]
    for thisMask in maskList:
        maskPath = maskDir + thisMask
        # Load Mask
        mask = tools.loadMask(maskPath)
        # mask = tools.loadMask(args.maskPath)

        # Load h5 Data, set up truth arrays
        dataDict = tools.loadH5Data(args.h5Dir, mask, sampleTypes, setTypes) 
        for scale in scales:

            # scales = [ "StandardScaler", "MinMax01Scaler", "MinMax11Scaler", "MaxAbsScaler", "RobustScaler", "QuantileTransformerNormal",  "QuantileTransformerUniform" ]
            # scales = [ "StandardScaler", "MinMax01Scaler"]
            # scales = [ "QuantileTransformerNormal", "QuantileUniform" ]
            # scales = [ "QuantileUniform" ]
            scales = [ "StandardScaler" ]
        
            for nodes in nodeList:
                beginloop = time.time()

                print("Begin " + scale)

                modelType = "tweakedOldBEST_BigChoke"
                suffix = str(nodes)
                modelFile, modelDir, maskSave, scalePath, plotDir, mySuffix = tools.dirStrings(
                                        modelType, maskPath, scale, args.year, args.outDir, suffix)


                # put this into dirStrings? make strings less clunky?
                # have mask path/save come out of load mask?
                
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
                # os.system('cp ' + args.maskPath + ' ' + maskSave)
                os.system('cp ' + maskPath + ' ' + maskSave)

                # BEST_model = train(modelFile, plotDir, mySuffix, float(args.patience), mask, scalePath, dataDict, nodes)
                # plotAll(BEST_model, args.h5Dir, plotDir, mySuffix, mask, modelType, scalePath)
                # del BEST_model
                
                train(modelFile, plotDir, mySuffix, float(args.patience), mask, scalePath, dataDict, nodes)
                plotAll(load_model(modelFile), args.h5Dir, plotDir, mySuffix, mask, modelType, scalePath)

                timelog = open("Logs/" + modelType + "_TimeLog.txt", "a") 
                timeTaken = divmod(time.time() - beginloop, 60.)
                loopMessage = "Ran " + mySuffix + ", took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
                print(loopMessage)
                timelog.write(loopMessage)
                timelog.close

        for key in dataDict.keys():
            del dataDict[key]
        del dataDict
# put NN train stuff in own script
# call it from this scriopt
# rename plotCM
# comment out that line tho
# have data loading and scaling happen outside of train script
# send the data in to train the model 
    # Check how long the script took to run
    timelog = open("Logs/" + modelType + "_TimeLog.txt", "a") 
    timeTaken = divmod(time.time() - startTime, 60.)
    timeMessage = "Finished script, total time was "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
    print(timeMessage)
    timelog.write(timeMessage)
    timelog.close
