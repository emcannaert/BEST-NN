#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# nnBEST.py ///////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Sam Abbott, Reyer Band, Johan S. Bonilla, Brendan Regnary,  ////////////
# This program trains BEST with flattened inputs //////////////////////////////////
# This uses the newBEST NN architecture ///////////////////////////////////////////
#==================================================================================

# user modules
import tools.functions as tools
startTime = tools.logTime() # Tracks how long script takes

# modules
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import tensorflow as tf
import shutil

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

# Print which gpu/cpu this is running on
sess = tf.Session(config=config)
h = tf.constant('hello world')
print(sess.run(h))

sampleTypes = ["Ht","Wb","Zt","QCD"]
setTypes = ["train", "validation", "test"]
years = ["2018"]
# years = ["2018"]
# years = ["2017", "2018", "2016_APV", "2016"]
# years = ["2016_APV", "2016", "2017"]

# maxEvents is the max number of events to pull from EACH of the 6 sample files
# "None" means use all of events in each file 
maxEvents = {"train":25000,      "validation":2500,   "test":2500}
# maxEvents = {"train":2000000, "validation":200000, "test":None}
# maxEvents = {"train":250000,  "validation":25000,  "test":None}
# maxEvents = {"train":50000,   "validation":10000,  "test":None}

def trainNNBEST(args, strings,mask, dataDict):
    print("Begin training BEST")
    
    # Unpack strings for readability
    modelFile = strings["modelFile"] 
    historyFile = strings["historyFile"] 
    # plotDir = strings["plotDir"] # not used in this function 
    suffix = strings["suffix"]

    # Unpack relevant args
    tol = float(args.tolerance)
    pat = float(args.patience)
    nodes = int(args.nodes)
    modelType = args.modelType

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
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combined)
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    combLayer   = Dense(nodes, kernel_initializer="glorot_normal", activation="relu"   )(combLayer)
    outputModel = Dense( 4, kernel_initializer="glorot_normal", activation="softmax")(combLayer)

    # Compile the model
    myModel = Model(inputs = [besModel.input], outputs = outputModel)
    myModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(myModel.summary() )

    # Early stopping
    # The smaller min_delta(tol) is, the longer the model with train, and the more accurate it will be.
    # A value of min_delta(tol) that is too small could possibly lead to overtraining...
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=tol, patience=pat, verbose=1, mode='auto', restore_best_weights=True)

    # Model checkpoint callback
    # This saves the model architecture + parameters into a .pb file
    model_checkpoint = ModelCheckpoint( modelFile, monitor='val_loss', 
                                        verbose=1, save_best_only=True,
                                        save_weights_only=False,
                                        period=1, mode='auto')

    # Train the neural network
    history = myModel.fit( [dataDict["trainEvents"]], dataDict["trainTruth"], 
                            validation_data = [[dataDict["validationEvents"]], dataDict["validationTruth"]], 
                            batch_size=BatchSize, epochs=200, 
                            callbacks=[early_stopping, model_checkpoint],
                            shuffle = True, steps_per_epoch=None
                            )

    print("Trained the neural network!")
    del dataDict

    # Record max accuracy
    accIndex = np.argmax(history.history['val_acc'])
    with open("Logs/" + modelType + "_accuracyLog.txt", "a") as f:
        f.write("Ran " + suffix + ": acc = " + str(history.history['acc'][accIndex])[:4] + ",  Max val_acc = " + str(history.history['val_acc'][accIndex])[:4] + '\n')

    dump(history.history, historyFile)

def prepareSavedModel(strings):
    import keras.backend as K
    K.set_learning_phase(0)
    
    # Unpack strings for readability
    modelFile = strings["modelFile"] 
    modelDir  = strings["modelDir"] 

    model = load_model(modelFile)
    print (model.inputs)
    print (model.outputs)

    graphDir = modelDir+'GraphExport/'
    buildDir = modelDir+'BuilderExport/'
    outTensor = model.outputs[0]
    outputs = tf.compat.as_text(outTensor.name)

    with K.get_session() as tempSess:
        # outputs = ["dense_20/Softmax"]
        constant_graph = tf.graph_util.convert_variables_to_constants(tempSess, tempSess.graph.as_graph_def(), [outputs[:-2]])
        tf.train.write_graph(constant_graph, graphDir, "constantgraph.pb", as_text=False)
        if os.path.exists(buildDir): shutil.rmtree(buildDir)
        builder = tf.saved_model.builder.SavedModelBuilder(buildDir)
        builder.add_meta_graph_and_variables(tempSess, [tf.saved_model.tag_constants.SERVING])
        builder.save()    
    del K
    print("Saved model as constant graph")

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to train neural network.')
    parser.add_argument('-hd','--h5Dir', dest='h5Dir',
                        default="/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/formatConverter/h5samplesSplit/",
                        help="Input File Dir [default: /uscms/home/bonillaj/nobackup/h5samples_ULv1/]")
    parser.add_argument('-o','--outDir', dest='outDir',
                        default="models/",
                        help="Output Dir where models are saved [default: models/]")
    parser.add_argument('-mp','--maskPath', dest='maskPath',
                        default = "/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/formatConverter/masks/BESvarList.txt",
                        help="Path to mask file [default: ../formatConverter/masks/BESTMask.txt]")
    parser.add_argument('-sf','--suffix', dest='suffix',
                        default="",
                        help="Suffix, used to uniquely identify model [default: '']")
    parser.add_argument('-sc','--scale', dest='scale',
                        default="flattened_standardized_maxAbs", # default="newBEST_Basic"
                        help="String used by MakeStandardInputs.py to name scaled data files [default: standardized]")
    parser.add_argument('-mt','--modelType', dest='modelType',
                        default="nnBEST",
                        help="Name of directory within models/ and plots/ [default: nnBEST]")
    parser.add_argument('-y','--years', dest='years',
                        default="2018",
                        help="Year of data taking to use [default: 2018]")
    parser.add_argument('-p','--patience', dest='patience',
                        default="20",
                        help="Number of Epochs to wait for improvement before EarlyStopping [default: 20]")
    parser.add_argument('-tol','--tolerance', dest='tolerance',
                        default="0.001",
                        help="Improvement tolerance for EarlyStopping; smaller tolerance means longer training [default: 0.01]")
    parser.add_argument('-n','--nodes', dest='nodes',
                        default="40",
                        help="Number of nodes per hidden layer [default: 83]")
    parser.add_argument('-r','--replace', dest='replace',
                        action='store_true',
                        help="Boolean, use flag to overwrite current model and plots [default: False]")
    parser.add_argument('-t','--train', dest='train',
                        action='store_false',
                        help="Boolean, use flag to load an already trained model and plot it. [default: True]")    
    args = parser.parse_args()

    if not args.years == "all": years = args.years.split(',')

    stringYearDict = {}
    for year in years:
        startTimeYear = tools.logTime() # Tracks how long this year takes

        # Generate appropriate helper strings, check dirs
        strings = tools.dirStrings(args, year)
        # stringYearDict[year] = strings

        # Load Mask
        mask, _ = tools.loadMask(args.maskPath)

        if args.train: # Run with -t to only plot the performance of the model
            # Load h5 Data, set up truth arrays
            dataDict = tools.loadH5Data(args, mask, sampleTypes, ["train", "validation"], maxEvents, year) 

            # Shuffle arrays
            tools.shuffleArray(dataDict)

            # Train using nnBEST
            trainNNBEST(args, strings, mask, dataDict)
            del dataDict

        # Load test data, evaluate model performance
        # dataDict = tools.loadH5Data(args, mask, sampleTypes, ["test"], maxEvents, year)
        dataDict = tools.loadH5Data(args, [True], sampleTypes, ["test"], maxEvents, year)

        # Load model
        print("Using BEST to predict...")
        # BESpredict  = load_model(strings["modelFile"]).predict(dataDict["testEvents"])
        BESpredict  = load_model(strings["modelFile"]).predict(dataDict["testEvents"][:,mask])

        # Make all performance plots
        tools.plotAll(args, strings, dataDict["testTruth"], args.modelType, BESpredict, year)
        # tools.plotAll(args, strings, dataDict, mask, year)
        # tools.plotAll(args, strings, year, "flattened")
        # tools.plotAll(args, strings, year, "flatTop")

        tools.logTime(startTimeYear, strings["suffix"])

    # # Save as .pb model for analysis
    # for year in years: prepareSavedModel(stringYearDict[year])
    
    tools.logTime(startTime)

