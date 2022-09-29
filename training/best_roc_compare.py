#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# best_roc_compare.py ////////////////////////////////////////////////////////
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
import tools.functions as tools

# enter batch mode in root (so python can access displays)
#root.gROOT.SetBatch(True)


sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
sampleFileTypesScaled = ["WW","ZZ","HH","TT","BB","QCD"]
sampleFileTypes = ["WW","ZZ","HH","TT","BB","QCD"]
samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]
# h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_OR/"
plotDir  = "plots/BEScompare/"
maskPath = "../formatConverter/h5samples/BESvarList.txt"
years = ["2016_APV", "2016", "2017", "2018"]

def plotROCyearcompare(setType="flattened"):

    # _, varDict = tools.loadMask(maskPath)    
    # r_varDict = {v:k for k,v in varDict.items()} #invert dictionary 

    modelDir  = "models/nnBEST/"

    rocDict = dict()

    for year in years:
        thisModel = setType+"_"+year
        # Load BES model and predict
        modelPath = os.path.join(modelDir,thisModel,"BEST_model_" + thisModel + ".h5")
        # modelPath = modelDir + "BEST_model_" + modelKey + ".h5"
        # modelPath = myModelDir + "BEST_model_" + modelKey + "_finalRuntime.h5"
        print(modelPath)
        model_BESonly = load_model(modelPath)
        # scalePath = "ScalerParameters_" + setType + "/BESTScalerParameters_" + year + ".txt"    

            # thisTest = thisModel + "_" + testSet
        # h5Path = "Sample_"+year+"_BESTinputs_test_" + setType + ".h5"
        h5Path = "Sample_"+year+"_BESTinputs_test_" + setType + "_standardized.h5"

        print("Loading data...")
        eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[()] for mySample in sampleFileTypesScaled]

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
                    label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
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
    eventArrays = [np.array(h5py.File(h5Dir + mySample + h5Path, "r")["BES_vars"])[()] for mySample in sampleFileTypesScaled]

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

def plotROCCompare(h5Dir, plotDir, samples):

    #==================================================================================
    # Load Test Data //////////////////////////////////////////////////////////////////
    #==================================================================================
    
    # BES variables network for comparison
    modelDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/"
    modelInfo = {
        # "300Basic_300_Z":["longLearn","Basic","300_Z"], "300Qmpxy_300_Z":["longLearn","Qmpxy","300_Z"], "300Qall_300_Z":["longLearn","Qall","300_Z"], 
        # "300Basic_300":["longLearn","Basic","300"], "300Basic_300_Z":["longLearn","Basic","300_Z"], "300Basic_300_B":["longLearn","Basic","300_B"], "300Basic_300_ZB":["longLearn","Basic","300_ZB"],
        # "300Basic_300_Z":["longLearn","Basic"], "300Qmpxy_300_Z":["longLearn","Qmpxy"], "300Qall_300_Z":["longLearn","Qall"], 
        # "300Basic_300":["longLearn","Basic"], "300Basic_300_Z":["longLearn","Basic"], "300Basic_300_B":["longLearn","Basic"], "300Basic_300_ZB":["longLearn","Basic"], 
       
       
        # "longBasic":["maskFix","Basic",""], 
        # "longBasic_Z":["maskFix","Basic","_Z"], "longBasic_B":["maskFix","Basic","_B"], "longBasic_ZB":["maskFix","Basic","_ZB"], 
        # "longBasic_ak8":["maskFix","Basic","_ak8"], "longBasic_ak8Z":["maskFix","Basic","_ak8Z"],

        # "140Basic300WHT400":["recheck_long","Basic","300WHT400"],
        # "140Basic300Wak8HT400":["recheck_long","Basic","300Wak8HT400"],
        # "140Basic300Wak8SDHT400":["recheck_long","Basic","300Wak8SDHT400"],


        "140Basic300WHT400":["recheck_long_2","Basic","300WHT400", "WHT300400"],
        "140Basic300Wak8HiggsTop400":["recheck_long_2","Basic","300Wak8HiggsTop400", "WHT300400ak8"],
        "140Basic300Wak8SDHT400":["recheck_long_2","Basic","300Wak8SDHT400", "WHT300400ak8SD"],

        "140Basic300Wbothak8HT400":["recheck_long","Basic","300Wbothak8HT400", "WHT300400bothak8"],
    }

    modelKeys = list(modelInfo.keys())
    modelKeys.sort()
    rocDict = dict()
    scale = "newBEST_Basic"
    dataDict = tools.loadH5Data(h5Dir, [True], sampleTypes, ["test"], scale) 


    for modelKey in modelKeys:
        print('\n'+modelKey)
        # modelType = "newBEST_" + modelInfo[modelKey][0] + "/"
        # scale     = "newBEST_" + modelInfo[modelKey][1]
        
        modelType = modelInfo[modelKey][0] + "/"
        scale     = modelInfo[modelKey][1]
        suffix    = modelInfo[modelKey][2]
        myModelDir = modelDir + modelType + modelKey + "/"

        # Load h5 data, set up truth arrays
        # mask, _ = tools.loadMask(myModelDir + "newBESTMask_" + suffix + ".txt")
        # mask, _ = tools.loadMask(myModelDir + "fixBESTMask" + suffix + ".txt")
        mask, _ = tools.loadMask(myModelDir + suffix + ".txt")
        # dataDict = tools.loadH5Data(h5Dir, mask, sampleTypes, ["test"], scale) 
        
        # Load BES model and predict
        modelPath = myModelDir + "BEST_model_" + modelKey + ".h5"
        # modelPath = myModelDir + "BEST_model_" + modelKey + "_finalRuntime.h5"
        print(modelPath)
        model_BESonly = load_model(modelPath)

        # model_BESonly = load_model(myModelDir + "BEST_model_" + modelKey + ".h5")
        # BESpredict = model_BESonly.predict([dataDict["testEvents"]])
        BESpredict = model_BESonly.predict([dataDict["testEvents"][:,mask]])
        # eventData = dataDict["testEvents"][:,mask]
        # print(model_BESonly.summary())
        # print(eventData.shape)
        # BESpredict = model_BESonly.predict([eventData])
        # print(BESpredict.shape, BESpredict[0])
        del model_BESonly
        # del eventData

        # Compute ROC curve and area for each class
        n_classes = len(samples) 
        fprBES = dict()
        tprBES = dict()
        roc_auc_BES = dict()
        for i, sample in enumerate(samples):
            fprBES[sample], tprBES[sample], _ = roc_curve(dataDict["testTruth"][:, i], BESpredict[:, i])
            roc_auc_BES[sample] = auc(fprBES[sample], tprBES[sample])

        # Compute micro-average ROC curve and ROC area
        fprBES["micro"], tprBES["micro"], _ = roc_curve(dataDict["testTruth"].ravel(), BESpredict.ravel() )
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
            lab = modelInfo[modelKey][3]
            plt.plot(fpr, tpr, 
                    # label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                    # label= modelKey + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
                    label= lab + ' ROC Curve (area = ' + str(roc_auc)[:6] + ') ',
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

# Run the ROC curve plot maker
# plotROCCompare(h5Dir, plotDir, samples)

# for year in years: plotROCflatcompare(year)

# plotROCyearcompare()

for year in years: plotROCpTcompare(year)
