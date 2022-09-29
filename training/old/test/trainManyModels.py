#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# oldBEST.py //////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Reyer Band, Johan S. Bonilla, Brendan Regnary, Sam Abbott ////////////
# This program trains BEST with flattened inputs //////////////////////////////////
# This uses the original oldBEST NN architecture //////////////////////////////////
#==================================================================================

################################## NOTES TO SELF ##################################
# Check for conistency, add comments.
# import time
# time.sleep(7200)

import tools.functions as tools
startTime = tools.logTime() # Tracks how long script takes

# modules
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import tensorflow as tf

# set up keras
import argparse, os
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" # must set backend before importing keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.externals.joblib import load, dump


# set up gpu environment
from keras import backend as k
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
k.tensorflow_backend.set_session(tf.Session(config=config))

# user modules
# import tools.functions as tools
from plotBESTPerformance import plotAll

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
# TrnValEvents = [ 500000, 50000]
# TrnValEvents = [ 250000, 25000]
# TrnValEvents = [ 125000, 12500]
# TrnValEvents = [ 50000, 10000]
TrnValEvents = [ 50000, 5000]

trainMaxEvents      = TrnValEvents[0]
validationMaxEvents = TrnValEvents[1]

setTypes = ["validation", "train"]

def prepInputData(mask, scale, dataDict, TrnValEvents):
    scaledDataDict = {}

    # Shuffle arrays
    # rng_state = np.random.get_state()
    # tools.shuffleArray(dataDict, rng_state)
    
    # Load scaler model  
    # scaler = tools.loadScalerModel(scale, mask)
    
    scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    scalePath = scaleDir + scale + ".joblib"
    if not os.path.isfile(scalePath):
        print(scalePath, "does not exist")
        quit()
    scaler = load(scalePath)
    print(scale, scaler.get_params())

    trainMaxEvents      = TrnValEvents[0]
    validationMaxEvents = TrnValEvents[1]

    print("Scaling validation...")
    # scaledValidEvents = scaler.transform(dataDict["validationEvents"][:validationMaxEvents])
    scaledDataDict["validationEvents"] = scaler.transform(dataDict["validationEvents"][:validationMaxEvents])
    # scaledDataDict["validationEvents"] = scaler.transform(dataDict["validationEvents"])
    print("Scaling train...")
    # scaledTrainEvents = scaler.transform(dataDict["trainEvents"][:trainMaxEvents])
    scaledDataDict["trainEvents"] = scaler.transform(dataDict["trainEvents"][:trainMaxEvents])
    # scaledDataDict["trainEvents"] = scaler.transform(dataDict["trainEvents"])
    print("Scaling test...")
    scaledDataDict["testEvents"] = scaler.transform(dataDict["testEvents"])
    del scaler

    # truthValid = dataDict["validationTruth"][:validationMaxEvents]
    # truthTrain = dataDict["trainTruth"][:trainMaxEvents]
    # scaledDataDict["validationTruth"] = dataDict["validationTruth"][:validationMaxEvents]
    # scaledDataDict["trainTruth"] = dataDict["trainTruth"][:trainMaxEvents]
    scaledDataDict["validationTruth"] = dataDict["validationTruth"]
    scaledDataDict["trainTruth"] = dataDict["trainTruth"]
    scaledDataDict["testTruth"] = dataDict["testTruth"]

    # print("Input validation shapes", scaledValidEvents.shape, truthValid.shape, truthValid[0] )
    # print("Input train shapes",      scaledTrainEvents.shape, truthTrain.shape, truthTrain[0] )
    print("Total validation shapes", scaledDataDict["validationEvents"].shape, scaledDataDict["validationTruth"].shape, scaledDataDict["validationTruth"][0] )
    print("Total train shapes",      scaledDataDict["trainEvents"].shape, scaledDataDict["trainTruth"].shape, scaledDataDict["trainTruth"][0] )
    print("Total test shapes",       scaledDataDict["testEvents"].shape, scaledDataDict["testTruth"].shape, scaledDataDict["testTruth"][0] )
    print("Batch Size: " + str(BatchSize) + ", Epochs: 50")
    
    return scaledDataDict

def trainBEST(modelDir, plotDir, suffix, userPatience, mask, dataDict, nodes, TrnValEvents):
    print("Begin training BEST")
    
    # Shuffle arrays
    # tools.shuffleArray(dataDict)
    
    #==================================================================================
    # Train the Neural Network ////////////////////////////////////////////////////////
    #==================================================================================
    modelFile    = modelDir + "BEST_model_" + suffix + ".h5"
    historyFile  = modelDir + "history_" + suffix + ".joblib"  
        
    BatchSize = 1200
    trainMaxEvents      = TrnValEvents[0]
    validationMaxEvents = TrnValEvents[1]

    # Create the BES framework
    # Train the neural network

    # Input variables, shape = number of 'True' entries in mask
    besInputs = Input( shape=(np.array(mask).sum(), ) )        
    besModel  = Model( inputs = besInputs, outputs = besInputs )
    print(besModel.output)   

    # Add BES variables to the network
    combined = besModel.output

    # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    outputModel = Dense( 6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # # The network architecture consists of 3 hidden layers with 40 nodes in each layer using a rectified-linear activation function.
    # combLayer   = Dense(80, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    # combLayer   = Dense(80, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # combLayer   = Dense(80, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    # outputModel = Dense( 6, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # Compile the model
    myModel = Model(inputs = [besModel.input], outputs = outputModel)
    myModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(myModel.summary() )

    # Early stopping
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=userPatience, verbose=1, mode='auto')#, restore_best_weights=True,)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=userPatience, verbose=1, mode='auto', restore_best_weights=True)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=userPatience, verbose=1, mode='auto', restore_best_weights=True)

    # Model checkpoint callback
    # This saves the model architecture + parameters into dense_model.h5
    model_checkpoint = ModelCheckpoint( modelFile, monitor='val_loss', 
                                        verbose=1, save_best_only=True,
                                        save_weights_only=False,
                                        period=1, mode='auto')

    scaledValidEvents = dataDict["validationEvents"][..., mask]
    scaledTrainEvents = dataDict["trainEvents"][..., mask]

    # scaledValidEvents = dataDict["validationEvents"][:validationMaxEvents, mask]
    # scaledTrainEvents = dataDict["trainEvents"][:trainMaxEvents, mask]

    # scaledValidEvents = dataDict["validationEvents"][:validationMaxEvents]
    # scaledTrainEvents = dataDict["trainEvents"][:trainMaxEvents]

    # scaledValidEvents = dataDict["validationEvents"][()]
    # scaledTrainEvents = dataDict["trainEvents"][()]


    truthValid = dataDict["validationTruth"][()]
    truthTrain = dataDict["trainTruth"][()]

    # truthValid = dataDict["validationTruth"][:validationMaxEvents]
    # truthTrain = dataDict["trainTruth"][:trainMaxEvents]

    print("Input validation shapes", scaledValidEvents.shape, truthValid.shape, truthValid[0] )
    print("Input train shapes",      scaledTrainEvents.shape, truthTrain.shape, truthTrain[0] )
    # print("Batch Size: " + str(BatchSize) + ", Epochs: 50")

    history = myModel.fit( [scaledTrainEvents], truthTrain, batch_size=BatchSize, 
                            # epochs=200, callbacks=[early_stopping, model_checkpoint],
                            epochs=200, callbacks=[model_checkpoint],
                            # epochs=100, callbacks=[model_checkpoint],
                            validation_data = [[scaledValidEvents], truthValid], 
                            shuffle = True, steps_per_epoch=None
                            )

    myModel.save(modelDir + "BEST_model_" + suffix + "_finalRuntime.h5")
    print("Trained the neural network!")
    del scaledTrainEvents
    del scaledValidEvents
    del truthTrain
    del truthValid
    del dataDict

    # Record max accuracy
    accIndex = np.argmax(history.history['val_acc'])
    with open("Logs/" + modelType + "_accuracyLog.txt", "a") as f:
        # accLog.write("Ran " + suffix + ", max acc = " + str(np.max(history.history['acc'])) + '\n')
        f.write("Ran " + suffix + ": acc = " + str(history.history['acc'][accIndex])[:6] + ",  Max val_acc = " + str(history.history['val_acc'][accIndex])[:6] + '\n')

    

    # Accuracy and Loss plots
    loss = [history.history['loss'], history.history['val_loss'] ]
    acc  = [history.history['acc'],  history.history['val_acc']  ]
    tools.plotPerformance(loss, acc, suffix, plotDir)
    dump(history.history, historyFile)

    print("Plotted BEST training Performance")


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
                        default="models/")
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
    parser.add_argument('-sc','--scale',
                        dest='scale',
                        # default="standardized")                        
                        default="newBEST_Basic")                        
    parser.add_argument('-r','--redoTraining', dest='redoTraining', default=False, action='store_true')
    args = parser.parse_args()

    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isdir(args.outDir):
        print(args.outDir, "does not exist")
        quit()

    scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    maskDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/"
    # maskDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/50/"

    maskDir += "indv/"
    maskList = [ 
                # "50GeV.txt", "100GeV.txt", "150GeV.txt", "200GeV.txt" 
                # "250GeV.txt", "300GeV.txt", "350GeV.txt", "400GeV.txt" 
                # "Higgs.txt", "Top.txt", "W.txt", "Z.txt" 
                # "ak8.txt", "ak8SoftDrop.txt", "Bottom.txt"
                # "Higgs.txt", "Top.txt", "300GeV.txt", "Bottom.txt"
                # "W.txt", "Z.txt", "WZ.txt" 
                # "ak8.txt", "ak8SoftDrop.txt", "bothAK8.txt"
                # "TopHiggs.txt", "TopW.txt", "300Top.txt" 
                # "TopHiggs.txt", "TopW.txt", "WHiggs.txt", "HiggsTopW.txt" 
                # "300Higgs.txt", "300W.txt", "300WHiggs.txt" 
                # "300Higgs.txt", "300W.txt", "WHiggs.txt", "300WHiggs.txt" 
                # "300Top.txt", "Topak8.txt", "300ak8.txt", "300Topak8.txt"
                # "frameInvariantVariables.txt"

                # "300Wak8.txt", "300Wak8SD.txt", "300Wbothak8.txt",
                # "300Wak8Higgs.txt", "300Wak8Top.txt", "300Wak8HiggsTop.txt",
                # "300Wak8SDHiggs.txt", "300Wak8SDTop.txt", "300Wak8SDHiggsTop.txt",
                # "300Wbothak8Higgs.txt", "300Wbothak8Top.txt", "300Wbothak8HiggsTop.txt",
                # "300Wak8Higgs400.txt", "300Wak8Top400.txt", "300Wak8HiggsTop400.txt",

                # try with SD, both, none. try with 250 350 instead of 300 400. try Z instead of W.
                # "300WHT400.txt", "300Wak8HiggsTop400.txt", "300Wak8SDHT400.txt", "300Wbothak8HT400.txt",
                # "300WHT400.txt", "300Wak8HiggsTop400.txt", "300Wak8SDHT400.txt",
                # "300WHT.txt", "300Wak8HT.txt", "300Wak8SDHT.txt", "300Wbothak8HT.txt",
                # "300WHT.txt", "300Wak8HT.txt", "300Wak8SDHT.txt", 
                # "300ZHT400.txt", "300Zak8HT400.txt", "300Zak8SDHT400.txt", "300Zbothak8HT400.txt",
                # "300ZHT400.txt", "300Zak8HT400.txt", "300Zak8SDHT400.txt",

                # "100200300400.txt", "100200300400ak8.txt",

                "300Wbothak8HT400.txt",


                ]    
    print(maskList)

    # do one for just the 50s

    # singleMask = tools.loadMask(maskDir + maskList[0])
    
    
    maskIndex = []
    varDict = {}
    maskDict = {}
    for mymask in maskList:
        maskDict[mymask] = [] 
        with open(maskDir + mymask, "r") as f:
            for line in f:
                # maskIndex.append(line.split(':')[0])
                index, var = line.split(':')
                var = var.strip()
                maskDict[mymask].append(index)
                if not index in varDict:
                    maskIndex.append(index)
                    varDict[index] = var

    maskIndex.sort(key=int)
    allMask = [True if str(i) in maskIndex else False for i in range(551) ]
    masks = {}
    
    # print(maskIndex)
    for mymask in maskList: 
        masks[mymask] = [True if index in maskDict[mymask] else False for index in maskIndex ]
        # print(masks[mymask])
    
    # quit()
    scale = "newBEST_Basic" 
    # scale =  "newBEST_Qmpxy"
    # scale =   "newBEST_Qall"

    modelType = "recheck_long_2_lowstats" 
    # modelType = "newBEST_maskFix" 
    # dataDict = tools.loadH5Data(args.h5Dir, singleMask, sampleTypes, ["train", "validation", "test"], scale) 
    dataDict = tools.loadH5Data(args.h5Dir, allMask, sampleTypes, ["train", "validation", "test"], scale) 
    # dataDict = tools.loadH5Data(args.h5Dir, [True], sampleTypes, ["train", "validation", "test"], scale) 
    # dataDict = tools.loadH5Data(args.h5Dir, [True], sampleTypes, ["validation"], scale) 

    # Shuffle arrays
    # rng_state = np.random.get_state()
    tools.shuffleArray(dataDict)

    # suffix = "long"
    # nodeList = [ 140, 120, 100 ]
    nodeList = [ 140 ]
    # nodeList = [ 100, 120, 140 ]
    # nodeList = [ 40, 60, 80, 100 ]
    # nodeList = [ 60, 80, 100 ]
    for nodes in nodeList:
        print("Begin " + str(nodes))
        nodeTime = tools.logTime()
        
        for thisMask in maskList:
            print("Begin " + thisMask)
            maskTime = tools.logTime()
            # if (thisMask == "ak8.txt") and (nodes == 60): continue
            # if (not thisMask == "Bottom.txt") and (nodes == 60): continue
            
            
            # mask = singleMask
            # Load Mask
            maskPath = maskDir + thisMask
            # mask = tools.loadMask(maskPath)
            # mask = tools.loadMask(args.maskPath)
            mask = masks[thisMask]
            print("Mask size: " + str(np.sum(mask)))

            modelDir, plotDir, suffix = tools.dirStrings(modelType, args, maskPath, str(nodes))

            print("Begin training new model...")
            trainBEST(modelDir, plotDir, suffix, float(args.patience), mask, dataDict, nodes, TrnValEvents)

            plotAll(load_model(modelDir + "BEST_model_" + suffix + ".h5"), dataDict, plotDir, suffix, modelType, mask)

            # Record how long mask took
            tools.logTime(maskTime, suffix)

        # Record how long nodes took
        tools.logTime(nodeTime, "Total " + str(nodes))

    del dataDict
    
    # Record how long total script took
    tools.logTime(startTime)
