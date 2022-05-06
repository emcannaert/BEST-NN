#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# combonationROCplotter.py ////////////////////////////////////////////////////////
#==================================================================================
# This program evaluates BEST: HH Event Shape Topology Indentification Algorithm 
#==================================================================================

# modules
#import ROOT as root
import os
import numpy
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
samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]
h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
plotDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/plots/BEScompare/"

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
        # mask = tools.loadMask(myModelDir + "newBESTMask_" + suffix + ".txt")
        # mask = tools.loadMask(myModelDir + "fixBESTMask" + suffix + ".txt")
        mask = tools.loadMask(myModelDir + suffix + ".txt")
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
        all_fprBES = numpy.unique(numpy.concatenate([fprBES[sample] for sample in samples]) )

        # Interpolate all roc curves
        mean_tprBES = numpy.zeros_like(all_fprBES)
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
plotROCCompare(h5Dir, plotDir, samples)
