#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# oldBEST.py //////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Reyer Band, Johan S. Bonilla, Brendan Regnary, Samantha Abbott ///////
# This program trains BEST with flattened inputs //////////////////////////////////
# This uses the original oldBEST NN architecture //////////////////////////////////
#==================================================================================

################################## NOTES TO SELF ##################################
# Check for conistency, add comments.


import math
import training.tools.functions_test as tools
startTime = tools.logTime() # Tracks how long script takes

# modules
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import tensorflow as tf
import h5py

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
print(tf.__version__)
# sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]

BATCH_SIZE = 100
# BATCH_SIZE = 10
H5DIR = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
SAMPLE_TYPES = ["WW","ZZ","HH","TT","BB","QCD"]
MASK = []
EPOCHS = 100
NUM_CLASSES = len(SAMPLE_TYPES)
# SUB_EPOCH_SIZE = 100000
# BATCHES_PER_EPOCH =  // BATCH_SIZE # 100

# TRAIN_FILES = [H5DIR + mySample + "Sample_2017_BESTinputs_train_flattened_newBEST_Basic.h5" for mySample in SAMPLE_TYPES]
# VAL_FILES = [H5DIR + mySample + "Sample_2017_BESTinputs_validation_flattened_newBEST_Basic.h5" for mySample in SAMPLE_TYPES]


TRAIN_FILES = [h5py.File(H5DIR + mySample + "Sample_2017_BESTinputs_train_flattened_newBEST_Basic.h5", 'r')["BES_vars"] for mySample in SAMPLE_TYPES]
VAL_FILES = [h5py.File(H5DIR + mySample + "Sample_2017_BESTinputs_validation_flattened_newBEST_Basic.h5", 'r')["BES_vars"] for mySample in SAMPLE_TYPES]

TRAIN_SIZES = [f.shape[0] for f in TRAIN_FILES]
VAL_SIZES = [f.shape[0] for f in VAL_FILES]

TRAIN_EVENTS = [np.arange(filesize) for filesize in TRAIN_SIZES]    
VAL_EVENTS = [np.arange(filesize) for filesize in VAL_SIZES]    


TRAIN_BATCHES = np.min(TRAIN_SIZES) // BATCH_SIZE 
VAL_BATCHES = np.min(VAL_SIZES) // BATCH_SIZE 

def dataGenerator(is_training=False):
    print("\ncalled generator")
    print(is_training)

    # if is_training: hfiles = [h5py.File(file, 'r')["BES_vars"] for file in TRAIN_FILES]
    # else:           hfiles = [h5py.File(file, 'r')["BES_vars"] for file in VAL_FILES]
    
    # filesizes = [f.shape[0] for f in hfiles]
    # batches = np.min(filesizes) // BATCH_SIZE
    # idx = [np.arange(filesize) for filesize in filesizes]    


    if is_training: 
        files = TRAIN_FILES
        sizes = TRAIN_SIZES
        batches = TRAIN_BATCHES
    else:
        files = VAL_FILES
        sizes = VAL_SIZES
        batches = VAL_BATCHES
    
    for epoch in range(EPOCHS):
        idx = [np.arange(filesize) for filesize in sizes]    
        for ind in idx: np.random.shuffle(ind)
    
        for batch in range(batches):
            if batch == 0: print(is_training, batch, epoch); print()
            # print("\nBATCH: " + str(batch) + " is_train: " + str(is_training) + "\n")
            # print(np.array(idx[0]).shape)
            # print(idx[0][:10])
            start = batch * BATCH_SIZE
            end = (batch+1) * BATCH_SIZE
            # if batch + 1 == batches: end = None
            # else:                    end = (batch+1) * batch_size

            events = []
            for inds in idx:
                ev = inds[start:end]
                ev.sort()
                events.append(list(ev))

            #==================================================================================
            # Load h5 BEST Data ///////////////////////////////////////////////////////////////
            #==================================================================================
            # print("test")
            # print(events)
            # data = np.concatenate( [f[events[i]][...,MASK] for i, f in enumerate(files)] )
            data = [np.array(f[events[i]][...,MASK] )for i, f in enumerate(files)] 
            # data = np.concatenate( [f[events[i],MASK] for i, f in enumerate(files)] )
            # print("test2")
            # Create the truth labels:
            # labels = np.concatenate( [np.full(BATCH_SIZE,i) for i in range(NUM_CLASSES)] )

            labels = [np.zeros((BATCH_SIZE, NUM_CLASSES)) for i in range(NUM_CLASSES)] 
            # labels = np.concatenate((tempLab[i][:,i] for i in range(NUM_CLASSES)) )
            # del tempLab

            # Arrays are filled with zeros. Now set 1's to record Truth particle info
            for i in range(NUM_CLASSES):
                labels[i][:,i] = 1

            # labels = np.concatenate(labels)

            shuffidx = []
            for i in range(NUM_CLASSES):
                sidx = list(range(BATCH_SIZE))
                if is_training: np.random.shuffle(sidx)
                shuffidx.append(sidx)
            # print(shuffidx)

            # print(data)
            # print("test3")
            # rng_state = np.random.get_state()
            # np.random.set_state(rng_state)
            # np.random.shuffle(data)
            # np.random.set_state(rng_state)
            # np.random.shuffle(labels)
            # print("test4")
            # data = np.concatenate([arr[shuffidx[i]] for i, arr in enumerate(data)])
            # labels = np.concatenate(lab[shuffidx[i]] for i, lab in enumerate(labels))
            # yield data, labels
            yield np.concatenate([arr[shuffidx[i]] for i, arr in enumerate(data)]), np.concatenate([lab[shuffidx[i]] for i, lab in enumerate(labels)])
            # yield data, tf.one_hot(labels, NUM_CLASSES)


def trainBEST(modelDir, plotDir, suffix, userPatience, mask, nodes):
    print("Begin training BEST")
    global MASK
    MASK = mask
    # Shuffle arrays
    # tools.shuffleArray(dataDict)
    
    #==================================================================================
    # Train the Neural Network ////////////////////////////////////////////////////////
    #==================================================================================
    modelFile    = modelDir + "BEST_model_" + suffix + ".h5"
    historyFile  = modelDir + "history_" + suffix + ".joblib"  
        
    # BatchSize = 1200
    # trainMaxEvents      = TrnValEvents[0]
    # validationMaxEvents = TrnValEvents[1]

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
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=userPatience, verbose=0, mode='auto')#, restore_best_weights=True,)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=userPatience, verbose=0, mode='auto')#, restore_best_weights=True,)

    # Model checkpoint callback
    # This saves the model architecture + parameters into dense_model.h5
    model_checkpoint = ModelCheckpoint( modelFile, monitor='val_loss', 
                                        verbose=1, save_best_only=True,
                                        save_weights_only=False,
                                        period=1, mode='auto')

    # scaledValidEvents = dataDict["validationEvents"][:validationMaxEvents, mask]
    # scaledTrainEvents = dataDict["trainEvents"][:trainMaxEvents, mask]

    # # scaledValidEvents = dataDict["validationEvents"][:validationMaxEvents]
    # # scaledTrainEvents = dataDict["trainEvents"][:trainMaxEvents]
    
    # truthValid = dataDict["validationTruth"][:validationMaxEvents]
    # truthTrain = dataDict["trainTruth"][:trainMaxEvents]


    # print("Input validation shapes", scaledValidEvents.shape, truthValid.shape, truthValid[0] )
    # print("Input train shapes",      scaledTrainEvents.shape, truthTrain.shape, truthTrain[0] )
    # print("Batch Size: " + str(BatchSize) + ", Epochs: 50")

    # history = myModel.fit( [scaledTrainEvents], truthTrain, batch_size=BatchSize, 
    #                         # epochs=200, callbacks=[early_stopping, model_checkpoint],
    #                         epochs=100, callbacks=[model_checkpoint],
    #                         validation_data = [[scaledValidEvents], truthValid] )

    # Create Datasets
    # buff = BATCH_SIZE * 6

    # # train_data = tf.data.Dataset.from_generator(
    # #         dataGenerator, args=[True],
    # #         output_types = (tf.float64, tf.int32),
    # #         output_shapes= (tf.TensorShape(buff), tf.TensorShape(buff))
    # #         )

    # # val_data = tf.data.Dataset.from_generator(
    # #         dataGenerator, 
    # #         output_types = (tf.float64, tf.int32),
    # #         output_shapes= (tf.TensorShape(buff), tf.TensorShape(buff))
    # #         )

    # train_data.shuffle(buff, reshuffle_each_iteration=True).repeat()
    # val_data.repeat()


    # print(tf.__version__)
    # print(vars(Model().fit))
    # print(inspect.getdoc(Model().fit_generator))
    history = myModel.fit_generator(  dataGenerator(True),
                            validation_data = dataGenerator(),
    # history = myModel.fit_generator(  BESSequence(TRAIN_FILES),
                            # validation_data = BESSequence(VAL_FILES),                            
                            epochs=EPOCHS, 
                            verbose=1,
                            # epochs=200, callbacks=[early_stopping, model_checkpoint],
                            callbacks=[model_checkpoint],
                            steps_per_epoch = TRAIN_BATCHES,
                            validation_steps = VAL_BATCHES,
                            # shuffle = True,
                            # workers=10,                            
                            # use_multiprocessing = True,
                            
    )



    myModel.save(modelDir + "BEST_model_" + suffix + "_finalRuntime.h5")
    print("Trained the neural network!")

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
                "ak8.txt", "ak8SoftDrop.txt", "bothAK8.txt"
                # "TopHiggs.txt", "TopW.txt", "300Top.txt" 

                ]    
    print(maskList)
    # scale = "newBEST_Basic" 
    # scale =  "newBEST_Qmpxy"
    # scale =   "newBEST_Qall"

    modelType = "recheck" 
    # modelType = "newBEST_maskFix" 
    # dataDict = tools.loadH5Data(args.h5Dir, [True], sampleTypes, ["train", "validation", "test"], scale) 
    # dataDict = tools.loadH5Data(args.h5Dir, [True], sampleTypes, ["validation"], scale) 

    # Shuffle arrays
    # rng_state = np.random.get_state()
    # tools.shuffleArray(dataDict)

    # suffix = "long"
    # nodeList = [ 40, 60, 80, 100 ]
    nodeList = [ 60, 80, 100 ]
    for nodes in nodeList:
        print("Begin " + str(nodes))
        nodeTime = tools.logTime()
        
        for thisMask in maskList:
            print("Begin " + thisMask)
            maskTime = tools.logTime()
            if (thisMask == "ak8.txt") and (nodes == 60): continue
            # Load Mask
            maskPath = maskDir + thisMask
            mask = tools.loadMask(maskPath)
            # MASK = tools.loadMask(maskPath)
            # mask = tools.loadMask(args.maskPath)

            modelDir, plotDir, suffix = tools.dirStrings(modelType, args, maskPath, str(nodes))

            print("Begin training new model...")
            trainBEST(modelDir, plotDir, suffix, float(args.patience), mask, nodes)

            # plotAll(load_model(modelDir + "BEST_model_" + suffix + ".h5"), dataDict, plotDir, suffix, modelType, mask)

            # Record how long mask took
            tools.logTime(maskTime, suffix)

        # Record how long nodes took
        tools.logTime(nodeTime, "Total " + str(nodes))

    # del dataDict
    
    # Record how long total script took
    tools.logTime(startTime)
