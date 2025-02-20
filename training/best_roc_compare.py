#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# best_roc_compare.py /////////////////////////////////////////////////////////////
#==================================================================================
# This program evaluates BEST: HH Event Shape Topology Indentification Algorithm 
#==================================================================================

# modules
#import ROOT as root
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf


# get stuff from modules
#from root_numpy import tree2array
from scipy import interp
from sklearn.metrics import roc_curve, auc

# set up keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow" #must set backend before importing keras
from keras.models import load_model
from keras.utils import to_categorical

# user modules
# import tools.functions as tools
import training.tools.functions_test as tools

# enter batch mode in root (so python can access displays)
#root.gROOT.SetBatch(True)


sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
sampleFileTypes = ["WW","ZZ","HH","TT","BB","QCD"]
samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]
# h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
# h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_OR/"
h5Dir = "../formatConverter/h5samples/"
plotDir  = "plots/BEScompare/"
maskPath = "../formatConverter/h5samples/BESvarList.txt"
years = ["2016_APV", "2016", "2017", "2018"]

def plotROCyearcompare(setType="flattened"):

    # _, varDict = tools.loadMask(maskPath)    
    # r_varDict = {v:k for k,v in varDict.items()} #invert dictionary 


    rocDict = dict()

    for year in years:
        modelDir  = os.path.join("models/nnBEST/",year)

        maskPath = os.path.join(modelDir,"BESvarList_nopt.txt")
        mask, _ = tools.loadMask(maskPath)

        # thisModel = setType+"_"+year
        thisModel = year
        # Load BES model and predict
        # modelPath = os.path.join(modelDir,thisModel,"BEST_model_" + thisModel + ".h5")
        # modelPath = modelDir + "BEST_model_" + modelKey + ".h5"
        # modelPath = myModelDir + "BEST_model_" + modelKey + "_finalRuntime.h5"
        modelPath = os.path.join(modelDir,"BEST_model_" + year + ".h5")
        print(modelPath)
        model_BESonly = load_model(modelPath)
        # scalePath = "ScalerParameters_" + setType + "/BESTScalerParameters_" + year + ".txt"    

            # thisTest = thisModel + "_" + testSet
        # h5Path = "Sample_"+year+"_BESTinputs_test_" + setType + ".h5"
        h5Path = "Sample_"+year+"_BESTinputs_test_" + setType + "_standardized.h5"

        print("Loading data...")
        # eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[()] for mySample in sampleFileTypes]
        eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[:,mask] for mySample in sampleFileTypes]

        print("My " + thisModel + " events shape:", [eventArrays[i].shape for i in range(len(eventArrays))])

        # Create Truth arrays; shape: N_events x 6
        truthArrays = [np.zeros( (len(eventArrays[i]), len(sampleTypes)) ) for i in range(len(eventArrays))] 

        # Arrays are filled with zeros. Now set 1's to record Truth particle info
        for i in range(len(sampleTypes)):
            truthArrays[i][:,i] = 1.

        print("My " + thisModel + " truth shape: ", [ truthArrays[i].shape for i in range(len(truthArrays)) ] )
        
        print("Concatenating...")
        eventArrays = np.concatenate(eventArrays)
        truthArrays  = np.concatenate(truthArrays)

        print("My " + thisModel + " concatenated event shape: ", eventArrays.shape)
        print("My " + thisModel + " concatenated truth shape: ", truthArrays.shape)

        # eventArrays = tools.manualScale(eventArrays, scalePath, r_varDict)
        
        print("Predicting...")
        BESpredict = model_BESonly.predict(eventArrays)

        print("Calculating ROC...")
        # Compute ROC curve and area for each class
        n_classes = len(samples) 
        fprBES = dict()
        tprBES = dict()
        roc_auc_BES = dict()
        for i, sample in enumerate(samples):
            fprBES[sample], tprBES[sample], _ = roc_curve(truthArrays[:, i], BESpredict[:, i])
            roc_auc_BES[sample] = auc(fprBES[sample], tprBES[sample])

        # Compute micro-average ROC curve and ROC area
        fprBES["micro"], tprBES["micro"], _ = roc_curve(truthArrays.ravel(), BESpredict.ravel() )
        # del dataDict
        del BESpredict
        roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

        # Compute macro-average ROC curve and ROC area:
        # First aggregate all false positive rates
        all_fprBES = np.unique(np.concatenate([fprBES[sample] for sample in samples]) )

        # Interpolate all roc curves
        mean_tprBES = np.zeros_like(all_fprBES)
        for sample in samples:
            mean_tprBES += interp(all_fprBES, fprBES[sample], tprBES[sample] )

        # Average and compute macro AUC
        mean_tprBES /= n_classes

        fprBES["macro"] = all_fprBES
        tprBES["macro"] = mean_tprBES
        roc_auc_BES["macro"] = auc(fprBES["macro"], tprBES["macro"])

        rocDict[thisModel] = { "fpr":fprBES, "tpr":tprBES, "auc":roc_auc_BES }

    # modelKeys = rocDict.keys()
    # modelKeys.sort()

    # rocKeys = rocDict[modelKeys[0]]["auc"].keys()
    rocKeys = rocDict[thisModel]["auc"].keys()
    # print(rocDict)
    # Fill dictionary with plot label information
    labelDict = {} # { key: [plot label, plot title, plot path name], ... }
    for key in rocKeys:
        if   key == "micro": labelDict[key] = ["Micro Average", "average_micro"]
        elif key == "macro": labelDict[key] = ["Macro Average", "average_macro"]
        else:                labelDict[key] = [key + " Category", key]

    # Plot ROC Curves
    print("Plotting...")
    saveDir = plotDir + 'flatFlatTop/'+year+"/"
    if not os.path.isdir(saveDir): os.makedirs(saveDir)
    # print(rocKeys)
    for key in rocKeys:
        # print(key)
        # Assign these for readability:
        title = labelDict[key][0] + " ROC Curve Comparison " + year 
        path  = saveDir + labelDict[key][1] + '_ROCplot'
        
        plt.figure(1)
        for modelKey, thisDict in rocDict.items():
            fpr = thisDict["fpr"][key]
            tpr = thisDict["tpr"][key]
            roc_auc = thisDict["auc"][key]

            plt.plot(fpr, tpr, 
                    # label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                    # label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                    label= "BEST " + modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                     linewidth=2)
        # range=(0.00001, 1.),
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(path + '.png')
        plt.savefig(path + '.pdf')

        # plt.yscale('log')
        # plt.title( title + "_yLogScale")
        # plt.show()
        # plt.savefig(saveDir + title + "_ylog.png")

        # plt.xscale('log')
        # plt.title( title + "_xyLogScale")
        # plt.show()
        # plt.savefig(saveDir + title + "_xylog.png")

        # plt.yscale('linear')
        # plt.title( title + "_xLogScale")
        # plt.legend(loc="upper left")
        # plt.show()
        # plt.savefig(saveDir + title + "_xlog.png")

        plt.clf()
        plt.close()

    print("Check out completed plots at:\n" + saveDir)

def plotROCpTcompare(year):

    # Load in data
    print("Loading data...")
    h5Path = "Sample_"+year+"_BESTinputs_test_flattened_standardized.h5"
    eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[()] for mySample in sampleFileTypes]

    print("My test events shape:", [eventArrays[i].shape for i in range(len(eventArrays))])

    # Create Truth arrays; shape: N_events x 6
    truthArrays = [np.zeros( (len(eventArrays[i]), len(sampleTypes)) ) for i in range(len(eventArrays))] 

    # Arrays are filled with zeros. Now set 1's to record Truth particle info
    for i in range(len(sampleTypes)):
        truthArrays[i][:,i] = 1.

    print("My test truth shape: ", [ truthArrays[i].shape for i in range(len(truthArrays)) ] )

    print("Concatenating...")
    eventArrays = np.concatenate(eventArrays)
    truthArrays = np.concatenate(truthArrays)

    print("My test concatenated event shape: ", eventArrays.shape)
    print("My test concatenated truth shape: ", truthArrays.shape)
    # load in full data for correct year
    # load in model, eval, do it again
    # plot



    # Load BES model and predict
    modelPath = os.path.join("models/nnBEST","flattened_"+year,"BEST_model_flattened_"+year+".h5")
    print(modelPath)
    model = load_model(modelPath)

    BESpredict = model.predict(eventArrays)
    del model

    modelPathnopt = os.path.join("models/nnBEST_nopt","flattened_"+year,"BEST_model_flattened_"+year+".h5")
    print(modelPathnopt)
    modelnopt = load_model(modelPathnopt)

    maskPathnopt = "../formatConverter/h5samples/BESvarList_nopt.txt"
    mask, _ = tools.loadMask(maskPathnopt)    
    BESpredictnopt = modelnopt.predict(eventArrays[:,mask])
    del modelnopt; del mask

    del eventArrays

    predictDict = {"BES":BESpredict, "BES_nopT":BESpredictnopt}
    rocDict = dict()
    for ptstring, predict in predictDict.items():
        # Compute ROC curve and area for each class
        n_classes = len(samples) 
        fprBES = dict()
        tprBES = dict()
        roc_auc_BES = dict()
        for i, sample in enumerate(samples):
            fprBES[sample], tprBES[sample], _ = roc_curve(truthArrays[:, i], predict[:, i])
            roc_auc_BES[sample] = auc(fprBES[sample], tprBES[sample])

        # Compute micro-average ROC curve and ROC area
        fprBES["micro"], tprBES["micro"], _ = roc_curve(truthArrays.ravel(), predict.ravel() )

        roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

        # Compute macro-average ROC curve and ROC area:
        # First aggregate all false positive rates
        all_fprBES = np.unique(np.concatenate([fprBES[sample] for sample in samples]) )

        # Interpolate all roc curves
        mean_tprBES = np.zeros_like(all_fprBES)
        for sample in samples:
            mean_tprBES += interp(all_fprBES, fprBES[sample], tprBES[sample] )

        # Average and compute macro AUC
        mean_tprBES /= n_classes

        fprBES["macro"] = all_fprBES
        tprBES["macro"] = mean_tprBES
        roc_auc_BES["macro"] = auc(fprBES["macro"], tprBES["macro"])

        rocDict[ptstring] = { "fpr":fprBES, "tpr":tprBES, "auc":roc_auc_BES }

    rocKeys = rocDict[ptstring]["auc"].keys()

    # Fill dictionary with plot label information
    labelDict = {} # { key: [plot label, plot title, plot path name], ... }
    for key in rocKeys:
        if   key == "micro": labelDict[key] = ["Micro Average", "average_micro"]
        elif key == "macro": labelDict[key] = ["Macro Average", "average_macro"]
        else:                labelDict[key] = [key + " Category", key]

    # Plot ROC Curves
    print("Plotting...")
    saveDir = plotDir + 'ptcompare/'+year+"/"
    if not os.path.isdir(saveDir): os.makedirs(saveDir)

    for key in rocKeys:
        # Assign these for readability:
        title = labelDict[key][0] + " ROC Curve Comparison " + year 
        path  = saveDir + labelDict[key][1] + '_ROCplot'
        
        plt.figure(1)
        for modelKey, thisDict in rocDict.items():
            fpr = thisDict["fpr"][key]
            tpr = thisDict["tpr"][key]
            roc_auc = thisDict["auc"][key]

            plt.plot(fpr, tpr, 
                    # label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                    # label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                    label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                     linewidth=2)
        # range=(0.00001, 1.),
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(path + '.png')

        plt.yscale('log')
        plt.title( title + "_yLogScale")
        plt.show()
        plt.savefig(saveDir + title + "_ylog.png")

        plt.xscale('log')
        plt.title( title + "_xyLogScale")
        plt.show()
        plt.savefig(saveDir + title + "_xylog.png")

        plt.yscale('linear')
        plt.title( title + "_xLogScale")
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig(saveDir + title + "_xlog.png")


        plt.close()

    print("Check out completed plots at:\n" + saveDir)

def plotROCflatcompare(year):

    _, varDict = tools.loadMask(maskPath)    
    r_varDict = {v:k for k,v in varDict.items()} #invert dictionary 
 
    testSets = ["flattened", "flatTop"]

    modelDir  = "models/nnBEST/"

    rocDict = dict()

    for modelSet in testSets:
        thisModel = modelSet+"_"+year
        # Load BES model and predict
        modelPath = os.path.join(modelDir,thisModel,"BEST_model_" + thisModel + ".h5")
        # modelPath = modelDir + "BEST_model_" + modelKey + ".h5"
        # modelPath = myModelDir + "BEST_model_" + modelKey + "_finalRuntime.h5"
        print(modelPath)
        model_BESonly = load_model(modelPath)
        scalePath = "ScalerParameters_" + modelSet + "/BESTScalerParameters_" + year + ".txt"    

        for testSet in testSets:
            thisTest = thisModel + "_" + testSet
            h5Path = "Sample_"+year+"_BESTinputs_test_" + testSet + ".h5"

            print("Loading data...")
            eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[()] for mySample in sampleFileTypes]

            print("My " + testSet + " events shape:", [eventArrays[i].shape for i in range(len(eventArrays))])

            # Create Truth arrays; shape: N_events x 6
            truthArrays = [np.zeros( (len(eventArrays[i]), len(sampleTypes)) ) for i in range(len(eventArrays))] 

            # Arrays are filled with zeros. Now set 1's to record Truth particle info
            for i in range(len(sampleTypes)):
                truthArrays[i][:,i] = 1.

            print("My " + testSet + " truth shape: ", [ truthArrays[i].shape for i in range(len(truthArrays)) ] )
            
            print("Concatenating...")
            eventArrays = np.concatenate(eventArrays)
            truthArrays  = np.concatenate(truthArrays)

            print("My " + testSet + " concatenated event shape: ", eventArrays.shape)
            print("My " + testSet + " concatenated truth shape: ", truthArrays.shape)

            eventArrays = tools.manualScale(eventArrays, scalePath, r_varDict)

            BESpredict = model_BESonly.predict(eventArrays)

            # Compute ROC curve and area for each class
            n_classes = len(samples) 
            fprBES = dict()
            tprBES = dict()
            roc_auc_BES = dict()
            for i, sample in enumerate(samples):
                fprBES[sample], tprBES[sample], _ = roc_curve(truthArrays[:, i], BESpredict[:, i])
                roc_auc_BES[sample] = auc(fprBES[sample], tprBES[sample])

            # Compute micro-average ROC curve and ROC area
            fprBES["micro"], tprBES["micro"], _ = roc_curve(truthArrays.ravel(), BESpredict.ravel() )
            # del dataDict
            del BESpredict
            roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

            # Compute macro-average ROC curve and ROC area:
            # First aggregate all false positive rates
            all_fprBES = np.unique(np.concatenate([fprBES[sample] for sample in samples]) )

            # Interpolate all roc curves
            mean_tprBES = np.zeros_like(all_fprBES)
            for sample in samples:
                mean_tprBES += interp(all_fprBES, fprBES[sample], tprBES[sample] )

            # Average and compute macro AUC
            mean_tprBES /= n_classes

            fprBES["macro"] = all_fprBES
            tprBES["macro"] = mean_tprBES
            roc_auc_BES["macro"] = auc(fprBES["macro"], tprBES["macro"])

            rocDict[thisTest] = { "fpr":fprBES, "tpr":tprBES, "auc":roc_auc_BES }

    # modelKeys = rocDict.keys()
    # modelKeys.sort()

    # rocKeys = rocDict[modelKeys[0]]["auc"].keys()
    rocKeys = rocDict[thisTest]["auc"].keys()
    # print(rocDict)
    # Fill dictionary with plot label information
    labelDict = {} # { key: [plot label, plot title, plot path name], ... }
    for key in rocKeys:
        if   key == "micro": labelDict[key] = ["Micro Average", "average_micro"]
        elif key == "macro": labelDict[key] = ["Macro Average", "average_macro"]
        else:                labelDict[key] = [key + " Category", key]

    # Plot ROC Curves
    print("Plotting...")
    saveDir = plotDir + 'flatFlatTop/'+year+"/"
    if not os.path.isdir(saveDir): os.makedirs(saveDir)
    # print(rocKeys)
    for key in rocKeys:
        # print(key)
        # Assign these for readability:
        title = labelDict[key][0] + " ROC Curve Comparison " + year 
        path  = saveDir + labelDict[key][1] + '_ROCplot'
        
        plt.figure(1)
        for modelKey, thisDict in rocDict.items():
            fpr = thisDict["fpr"][key]
            tpr = thisDict["tpr"][key]
            roc_auc = thisDict["auc"][key]

            plt.plot(fpr, tpr, 
                    # label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                    # label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                    label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                     linewidth=2)
        # range=(0.00001, 1.),
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(path + '.png')

        plt.yscale('log')
        plt.title( title + "_yLogScale")
        plt.show()
        plt.savefig(saveDir + title + "_ylog.png")

        plt.xscale('log')
        plt.title( title + "_xyLogScale")
        plt.show()
        plt.savefig(saveDir + title + "_xylog.png")

        plt.yscale('linear')
        plt.title( title + "_xLogScale")
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig(saveDir + title + "_xlog.png")


        plt.close()

    print("Check out completed plots at:\n" + saveDir)

def getScores():
    print("\nGetting tagger scores...")
    #==================================================================================
    # Load Scores From Other Taggers //////////////////////////////////////////////////
    #==================================================================================
    
    year = "2017"
    h5Path = "Sample_"+year+"_BESTinputs_test_flattened_standardized.h5"

    maskPath = h5Dir + "BESvarList_scores.txt"
    mask, _ = tools.loadMask(maskPath)    

    print("Loading data...")
    print(h5Path)
    eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[:,mask] for mySample in sampleFileTypes]

    print("My test events shape:", [eventArrays[i].shape for i in range(len(eventArrays))])

    print("Concatenating...")
    eventArrays = np.concatenate(eventArrays)

    print("My test concatenated event shape: ", eventArrays.shape)

    # ParticleNet: First 17 vars [0:17]
    particleNetScores = eventArrays[:,0:17]

    # DeepAK8MD: Next 15 vars [17:32] 
    deepAK8MDScores = eventArrays[:,17:32]

    # DeepAK8: Final 15 vars [32:]
    deepAK8Scores = eventArrays[:,32:]

    del eventArrays

    print("ParticleNet shape: ", particleNetScores.shape)
    print("deepAK8MD shape: ", deepAK8MDScores.shape)
    print("deepAK8 shape: ", deepAK8Scores.shape)

    # Set up map for which score corresponds to which category
    # W=0, Z=1, H=2, T=3, B=4, QCD=5
    scoreDict = dict()
    scoreDict["ParticleNet"] = {0:2, 1:2, 2:2, 3:5, 4:5, 5:5, 6:5, 7:5, 8:3, 9:3, 10:3, 11:3, 12:0, 13:0, 14:1, 15:1, 16:1}
    scoreDict["deepAK8MD"]   = {0:2, 1:2, 2:2, 3:5, 4:5, 5:5, 6:5, 7:5, 8:3, 9:3, 10:0, 11:0, 12:1, 13:1, 14:1}
    scoreDict["deepAK8"]     = {0:2, 1:2, 2:2, 3:5, 4:5, 5:5, 6:5, 7:5, 8:3, 9:3, 10:0, 11:0, 12:1, 13:1, 14:1}

    # Get predictions, convert to usable form
    particleNetPredict = np.argmax(particleNetScores, axis=1)
    particleNetPredict = [scoreDict["ParticleNet"][score] for score in particleNetPredict]
    particleNetPredict = np.eye(6)[particleNetPredict]

    deepAK8MDPredict = np.argmax(deepAK8MDScores, axis=1)
    deepAK8MDPredict = [scoreDict["deepAK8MD"][score] for score in deepAK8MDPredict]
    deepAK8MDPredict = np.eye(6)[deepAK8MDPredict]

    deepAK8Predict = np.argmax(deepAK8Scores, axis=1)
    deepAK8Predict = [scoreDict["deepAK8"][score] for score in deepAK8Predict]
    deepAK8Predict = np.eye(6)[deepAK8Predict]

    print("ParticleNet predict shape: ", particleNetPredict.shape)
    print("deepAK8MD predict shape: ", deepAK8MDPredict.shape)
    print("deepAK8 predict shape: ", deepAK8Predict.shape)

    # These now match the form of BESPredict
    # Can use to plot ROC curves now
    return {"particleNet":particleNetPredict, "deepAK8MD":deepAK8MDPredict, "deepAK8":deepAK8Predict}


def plotROCCompare():

    #==================================================================================
    # Load Test Data //////////////////////////////////////////////////////////////////
    #==================================================================================
    
    # BES variables network for comparison
    # modelDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/"
    modelDir  = "models/"
    modelInfo = {
        "noBDisc":["noBDisc","standardized","flattened_2017", "BEST w/ no bDisc"],
        "deepCSV":["deepCSV","standardized","flattened_2017", "BEST w/ deepCSV"],
        "naive":["naive","standardized","flattened_2017", "BEST w/ deepJet"],
        # "advanced":["advanced","standardized","flattened_2017", "advanced"],
        "deepCSVnaive":["deepCSVnaive","standardized","flattened_2017", "BEST w/ deepJet & deepCSV"],
        
        # "2017":["nnBEST","standardized","2017", "BEST"],

        # "particleNetScores":["particleNet","","", "ParticleNet"],
        # "deepAK8MDScores":["deepAK8MD","","", "DeepAK8 Mass Decorrelated"],
        # "deepAK8Scores":["deepAK8","","", "DeepAK8"],
    }

    modelKeys = list(modelInfo.keys())
    modelKeys.sort()

    # Grab scores from other taggers if comparing
    if "Scores" in '\t'.join(modelKeys): scoreDict = getScores()

    rocDict = dict()
    year = "2017"
    h5Path = "Sample_"+year+"_BESTinputs_test_flattened_standardized.h5"

    print("Loading data...")
    print(h5Path)
    eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[()] for mySample in sampleFileTypes]

    print("My test events shape:", [eventArrays[i].shape for i in range(len(eventArrays))])

    # Create Truth arrays; shape: N_events x 6
    truthArrays = [np.zeros( (len(eventArrays[i]), len(sampleTypes)) ) for i in range(len(eventArrays))] 

    # Arrays are filled with zeros. Now set 1's to record Truth particle info
    for i in range(len(sampleTypes)):
        truthArrays[i][:,i] = 1.

    print("My test truth shape: ", [ truthArrays[i].shape for i in range(len(truthArrays)) ] )
    
    print("Concatenating...")
    eventArrays = np.concatenate(eventArrays)
    truthArrays  = np.concatenate(truthArrays)

    print("My test concatenated event shape: ", eventArrays.shape)
    print("My test concatenated truth shape: ", truthArrays.shape)


    for modelKey in modelKeys:
        print('\n'+modelKey)
        
        if "Scores" in modelKey: # Get Scores from other taggers
            tagger = modelInfo[modelKey][0]
            BESpredict = scoreDict[tagger]
        else: # Get predictions from a trained BEST model
            modelType = modelInfo[modelKey][0] + "/"
            scale     = modelInfo[modelKey][1]
            suffix    = modelInfo[modelKey][2]
            # myModelDir = modelDir + modelType + modelKey + "/"
            myModelDir = modelDir + modelType + suffix + "/"

            # Load h5 data, set up truth arrays
            # mask, _ = tools.loadMask(myModelDir + suffix + ".txt")
            maskPath = myModelDir + "BESvarList_" + modelKey + ".txt"
            # maskPath = myModelDir + "BESvarList_nopt.txt"
            mask, _ = tools.loadMask(maskPath)
            
            # Load BES model and predict
            # modelPath = myModelDir + "BEST_model_" + modelKey + ".h5"
            modelPath = myModelDir + "BEST_model_" + suffix + ".h5"
            print(modelPath)
            model_BESonly = load_model(modelPath)

            BESpredict = model_BESonly.predict([eventArrays[:,mask]])
            del model_BESonly

        print(BESpredict.shape)
        # Compute ROC curve and area for each class
        n_classes = len(samples) 
        fprBES = dict()
        tprBES = dict()
        roc_auc_BES = dict()
        for i, sample in enumerate(samples):
            fprBES[sample], tprBES[sample], _ = roc_curve(truthArrays[:, i], BESpredict[:, i])
            roc_auc_BES[sample] = auc(fprBES[sample], tprBES[sample])

        # Compute micro-average ROC curve and ROC area
        fprBES["micro"], tprBES["micro"], _ = roc_curve(truthArrays.ravel(), BESpredict.ravel() )
        # del dataDict
        del BESpredict
        roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

        # Compute macro-average ROC curve and ROC area:
        # First aggregate all false positive rates
        all_fprBES = np.unique(np.concatenate([fprBES[sample] for sample in samples]) )

        # Interpolate all roc curves
        mean_tprBES = np.zeros_like(all_fprBES)
        for sample in samples:
            mean_tprBES += interp(all_fprBES, fprBES[sample], tprBES[sample] )

        # Average and compute macro AUC
        mean_tprBES /= n_classes

        fprBES["macro"] = all_fprBES
        tprBES["macro"] = mean_tprBES
        roc_auc_BES["macro"] = auc(fprBES["macro"], tprBES["macro"])

        rocDict[modelKey] = { "fpr":fprBES, "tpr":tprBES, "auc":roc_auc_BES }

    rocKeys = rocDict[modelKeys[0]]["auc"].keys()
    # print(rocDict)
    # Fill dictionary with plot label information
    labelDict = {} # { key: [plot label, plot title, plot path name], ... }
    for key in rocKeys:
        if   key == "micro": labelDict[key] = ["Micro Average", "average_micro"]
        elif key == "macro": labelDict[key] = ["Macro Average", "average_macro"]
        else:                labelDict[key] = [key + " Category", key]

    # Plot ROC Curves
    saveDir = plotDir + '_'.join(modelKeys) + '/'
    if not os.path.isdir(saveDir): os.makedirs(saveDir)
    # print(rocKeys)
    print("Plotting...")
    for key in rocKeys:
        # print(key)
        # Assign these for readability:
        title = labelDict[key][0] + " ROC Curve Comparison" 
        path  = saveDir + labelDict[key][1] + '_ROCplot'
        
        plt.figure(1)
        for modelKey, thisDict in rocDict.items():
            fpr = thisDict["fpr"][key]
            tpr = thisDict["tpr"][key]
            roc_auc = thisDict["auc"][key]
            label = modelInfo[modelKey][3]
            plt.plot(fpr, tpr, 
                    # label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                    # label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                    label= label + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                     linewidth=2)
        # range=(0.00001, 1.),
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        # workPoints = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
        # for i, wp in enumerate(workPoints): 
        #     plt.axvline(wp, linestyle=':', color="red")
        #     plt.annotate(str(wp*100)+"%", [wp,0.1*(i+1)], color="black")


        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(path + '.png')
        plt.savefig(path + '.pdf')

        # plt.yscale('log')
        # plt.title( title + "_yLogScale")
        # plt.show()
        # plt.savefig(saveDir + title + "_ylog.png")

        # plt.xscale('log')
        # plt.title( title + "_xyLogScale")
        # plt.show()
        # plt.savefig(saveDir + title + "_xylog.png")

        # plt.yscale('linear')
        # plt.title( title + "_xLogScale")
        # plt.legend(loc="upper left")
        # plt.show()
        # plt.savefig(saveDir + title + "_xlog.png")


        plt.close()

    print("Check out completed plots at:\n" + saveDir)


# Run the ROC curve plot maker
plotROCCompare()

# for year in years: plotROCflatcompare(year)

# plotROCyearcompare()

# for year in years: plotROCpTcompare(year)
