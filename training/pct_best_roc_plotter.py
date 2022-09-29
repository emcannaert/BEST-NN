#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# combonationROCplotter.py ////////////////////////////////////////////////////////
#==================================================================================
# This program evaluates BEST: HH Event Shape Topology Indentification Algorithm 
#==================================================================================

# modules
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt
import tensorflow as tf

# get stuff from modules
from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix

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
plotDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/plots/PCTcompare/"

# PCT Networks
# file_PCT = h5py.File('/uscms/home/bonillaj/nobackup/h5samples_PCTv2/PCTLabFrameOnly13FeaturesLogInputs.h5', 'r')
# PCTFiles = [ h5py.File('/uscms/home/bonillaj/nobackup/h5samples_PCTv2/FullBESTPCT8FeaturesNoLog.h5', 'r'), 
# PCTFiles = [ h5py.File('/uscms/home/bonillaj/nobackup/models/PCTevaluations/PCTLabFrameOnly13FeaturesLogInputs100PFcands.h5', 'r'), 
#             h5py.File('/uscms/home/bonillaj/nobackup/models/PCTevaluations/PCTLabFrameOnly13FeaturesLogInputs50PFcands.h5', 'r')
#            ]
# PCTFiles = [ h5py.File('/uscms/home/bonillaj/nobackup/models/PCTevaluations/PCTLabFrameOnly13FeaturesLogInputs50PFcands.h5', 'r'), 
#              h5py.File('/uscms/home/bonillaj/nobackup/models/PCTevaluations/PCTLabFrameOnly13FeaturesLogInputs50PFcandsDecayStep1000000.h5', 'r'),
#              h5py.File('/uscms/home/bonillaj/nobackup/models/PCTevaluations/PCTLabFrameOnly13FeaturesLogInputs50PFcandsDecayStep10000000.h5', 'r')
#            ]
# PCTlegendNames = ['50 cand PCT', 'Decay Step 1000000', 'Decay Step 10000000' ]

PCTDir = "/uscms/home/msabbott/nobackup/PCT_HEP/"
# PCTnames = [ "1kLab", "10kLab", "500kLab" ]
# PCTnames = [ "1kLabSRRNB_1", "1kLabSRRNBLR4_1", "1kLabSRRNBLR5_1", "1kLabSRRNBLR6_1" ]
# PCTnames = [ "100kLabSRRNB_1", "100kLabSRRNBLR4_1", "100kLabSRRNBLR5_1", "100kLabSRRNBLR6_1" ]
# PCTnames = [ "10kLabSRRNB_2", "10kLabSRRNBLR4_1", "10kLabSRRNBLR5_1", "10kLabSRRNBLR6_1" ]
# PCTnames = [ "1kLabSRRNB_1", "10kLabSRRNB_1", "10kLabSRRNB_2", "100kLabSRRNB_1" ]
# PCTnames = [ "1kLab_1", "10kLab_1", "500kLab_E4", "2MLab_E2", "allStatsLab_E0" ]
# PCTnames = [ "500kLab_E4", "500kLab_E39", "2MLab_E2", "2MLab_E14", "allStatsLab_E0", "allStatsLab_E8" ]
# PCTnames = [ "500kLab_E39", "500kLabSR_E49", "500kLabSRRNB_1", "500kLabSRRNBLR6_1",]
# PCTnames = [ "1kSA", "10kSA", "100kSA", "allSA" ]

# PCTnames = [ "1kSARR", "1kSARRR", "10kSARR", "10kSARRR", "100kSARR", "100kSARRR_E25", "allSARR" ]
# PCTnames = [ "1kARR", "1kARRR", "10kARR", "10kARRR", "100kARR", "100kARRR_E21", "allARR" ]
# PCTnames = [ "1kSARRR", "1kARRR", "10kSARRR", "10kARRR", "100kSARRR", "100kARRR" ]
# PCTnames = [ "100kS_LV", "100kS_LVH" ]
# PCTnames = [ "100kSARRR", "100kS_LV_E31", "100kS_LVH_E25" ]

# PCTnames = [ "100kS_L_E25", "100kS_LV_E25", "100kS_LH_E25", "100kS_LVH_E25" ]

# PCTnames = [ "100kS_L_E25", "100kS_LH_E25", "100kS_LH" ]

# PCTnames = [ "100kkS_L_E22", "100kS_LV_E22", "100kS_LH_E22", "100kS_LVH_E22" ]
# PCTnames = [ "100kkS_L_E22", "100kS_LH_E22", "100kS_LT_E22", "100kS_LW_E22", "100kS_LZ_E22", "100kS_LB_E22", "100kS_LHWZT_E22", ]
# PCTnames = [ "100kkS_L_E22", "100kS_LE_E22", "100kS_LH_E22", "100kS_LT_E22", "100kS_LW_E22", "100kS_LZ_E22", "100kS_LB_E22", "100kS_LHWZT_E22", ]

# PCTnames = [ "100kS_L_E25", "100kkS_L_E25", ]

# PCTnames = [ "100kS_LE_E0", "100kS_LE_E35", ]
# PCTnames = [ "100kS_LHE", "100kS_LBE", "100kS_LTE", "100kS_LZE", "100kS_LWE", "100kSF_LBE", "100kSF_LTE", "100kS_LHWZTE", "100kSmix_LHWZTE" ]

PCTnames = [  "100kS_LBE", "100kS_LTE", "100kS_LWE" ]


# PCTnames = [ "1kSARR", "1kARR", "10kSARR", "10kARR", "100kSARR", "100kARR" ]
# PCTnames = [ "1kSARR", "1kSARRR", "10kSARR", "10kSARRR" ]
# PCTnames = [ "1kLab_1", "10kLab_1", "100kLab_1" ]
# PCTnames = [ "1kLab_1", "1kLabS_1", "1kLabR_1", "1kLabR_2" ]
# PCTnames = [ "1kLabSR_1", "1kSBB_1", "1kSBB2_1", "10kLabSR_1", "10kSBB2_1", "100kLabSR_1", "100kSBB2_1" ]
# PCTnames = [ "10kLabSR_1", "10kSBB2_1" ]
# PCTnames = [ "1kLabSR_1", "1kSRT_1", "1kST_1", "1kSI_1", "1kSBB_1", "1kSBB2_1", "1kSL1_1", "1kSL2_1"  ]
# PCTnames = [ "10kLabSR_1", "10kSB64_1", "10kSL_1", "10kST_1", "10kSI_1", "10kSD_1", "10kSC_1" ]
# PCTnames = [ "100kLabSR_1", "100kSB64_1", "100kSL_1", "100kST_1", "100kSI_1", "100kSD_1", "100kSC_1" ]
# PCTnames = [ "10kLab_1", "10kLabR_1", "10kLabS_1", "10kLabSR_1" ]
# PCTnames = [ "10kLabS_1", "10kLabS_2" ]
# PCTnames = [ "1kLabS_1", "1kLabS_2", "1kLabS_3" ]
# PCTnames = [ "1kLab_1", "1kLabS_1" ]
# PCTnames = [ "10kLab_1", "10kLab_2", "10kLab_3" ]
# PCTnames = [ "100kLab_1", "100kLabS_1", "100kLabD_1" ]


# PCTFiles = [h5py.File(PCTDir + 'h5/' + f + '.h5', 'r') for f in PCTnames ]
# PCTFiles = [h5py.File(PCTDir + 'h5/22/' + f + '.h5', 'r') for f in PCTnames ]
PCTlegendNames = [ 'PCT_' + f  for f in PCTnames ]

PCTcolors = [ 'green', 'blue', 'red']
def pctLossPlotter(PCTnames, saveDir):
    # text file is a bit messy. always has 'mean loss: ' before the loss. 
    # alternates between loss and val loss, so hacky solution:
    print("Plotting Loss Curves...")
    lossDict = {}
    lossDir = saveDir + "loss/"
    if not os.path.isdir(lossDir): os.makedirs(lossDir)

    accDict = {}
    accDir = saveDir + "acc/"
    if not os.path.isdir(accDir): os.makedirs(accDir)    

    for i, name in enumerate(PCTnames):
        # if name == "100kS_LH_E25": continue
        print(name)

        # allLosses = []
        # if 'E' in name: logPath = PCTDir + 'logs/' + name[:name.rfind('_')] + '/log_train.txt'
        # else:           logPath = PCTDir + 'logs/' + name + '/log_train.txt'
        # with open(logPath, 'r') as file:
        #      for line in file:
        #          if "mean loss:" in line: allLosses.append(float(line[11:]))
        # loss = []
        # val_loss = []
        # for i, thisLoss in enumerate(allLosses):
        #     if i % 2 == 0: loss.append(thisLoss)
        #     else:          val_loss.append(thisLoss)
        # # print(len(allLosses), len(loss), len(val_loss))
        # # print(allLosses, loss, val_loss)


        loss = []; val_loss = []
        acc = []; val_acc = []
        if name == "100kS_L_E25": logPath = PCTDir + 'logs/100kSARRR/log_train.txt'
        elif name == "100kS_L": logPath = PCTDir + 'logs/100kSARRR/log_train.txt'
        # elif 'E' in name: logPath = PCTDir + 'logs/' + name[:name.rfind('_')] + '/log_train.txt'
        else:           logPath = PCTDir + 'logs/' + name + '/log_train.txt'
        with open(logPath, 'r') as file:
            for line in file:
                if   "Train Mean Loss:" in line: loss.append(float(line[17:]))
                elif "Train Loss:"      in line: loss.append(float(line[12:]))
                elif "Train Acc:"      in line: acc.append(float(line[11:]))
                elif "Sub"              in line: continue
                elif "Eval Mean Loss:"  in line: val_loss.append(float(line[16:]))
                elif "Val Loss:"        in line: val_loss.append(float(line[10:]))
                elif "Val Acc:"        in line: val_acc.append(float(line[9:]))

        # if name == "100kSARRR":
        # loss = loss[:23]; val_loss = val_loss[:23]
        # acc  =  acc[:23]; val_acc  =  val_acc[:23]

        # lossDir += "trim_"
        # print(lossDir, len(loss), len(val_loss))

        lossDict[name] = [loss,val_loss]

        # Plot Loss vs. Epoch
        plt.figure()
        plt.plot(loss,     label='loss; Min loss: ' + str(np.min(loss))[:6] + ', Epoch: ' + str(np.argmin(loss)) )
        plt.plot(val_loss, label='val_loss; Min val_loss: ' + str(np.min(val_loss))[:6] + ', Epoch: ' + str(np.argmin(val_loss)) )
        plt.title(name + " loss and val_loss vs. epochs")
        plt.legend(loc="upper right")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.savefig(lossDir+name+"_loss.pdf")
        plt.savefig(lossDir+name+"_loss.png")
        plt.clf()

        if len(acc) == 0: continue
        accDict[name]  = [acc,val_acc]
        # 100kS_LHWZT, 100kS_LH

        # Plot Acc vs. Epoch
        plt.plot(acc,     label='acc; Max acc: ' + str(np.max(acc))[:6] + ', Epoch: ' + str(np.argmax(acc)) )
        plt.plot(val_acc, label='val_acc; Max val_acc: ' + str(np.max(val_acc))[:6] + ', Epoch: ' + str(np.argmax(val_acc)) )
        plt.title(name + " acc and val_acc vs. epochs")
        plt.legend(loc="lower right")
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        # plt.savefig(accDir+name+"_acc.pdf")
        plt.savefig(accDir+name+"_acc.png")
        plt.close()

        # # Plot Normalized Losses vs. Epoch
        # loss_auc = np.trapz(loss)
        # val_loss_auc = np.trapz(val_loss)
        # print(name, val_loss_auc/loss_auc)
        # plt.figure()
        # plt.plot()
        # plt.plot(loss/loss_auc,         
        #         label='Min loss: ' + str(np.min(loss))[:6] + ', Epoch: ' + str(np.argmin(loss)) + ', AUC: ' + str(loss_auc)[:6] )
        # plt.plot(val_loss/val_loss_auc, 
        #         label='val_loss; Min val_loss: ' + str(np.min(val_loss))[:6] + ', Epoch: ' + str(np.argmin(val_loss))  + ', AUC: ' + str(val_loss_auc)[:6] )
        # plt.title("Normalized " + name + " loss and val_loss vs. epochs")
        # plt.legend(loc="upper right")
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # # plt.savefig(lossDir+name+"_loss.pdf")
        # plt.savefig(lossDir+name+"_loss_normalized.png")
        # plt.close()

    # Plot Loss vs. Epoch
    # plt.figure()
    plt.figure(figsize=(12.8,9.6))
    plt.rcParams.update({'font.size': 18})

    for i, name in enumerate(PCTnames):
        loss = lossDict[name][0]
        plt.plot(loss, label=name+' loss; Min loss: ' + str(np.min(loss))[:6] + ', Epoch: ' + str(np.argmin(loss)) )
    plt.title("Loss vs. epochs")
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(lossDir+"loss_compare.png")
    plt.clf()

    for i, name in enumerate(PCTnames):
        val_loss = lossDict[name][1]
        plt.plot(val_loss, label=name+' val_loss; Min val_loss: ' + str(np.min(val_loss))[:6] + ', Epoch: ' + str(np.argmin(val_loss)) )
    plt.title("Val_loss vs. epochs")
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(lossDir+"val_loss_compare.png")
    plt.clf()

    # Plot Acc vs. Epoch
    for i, name in enumerate(PCTnames):
        if not name in accDict: continue
        acc = accDict[name][0]
        plt.plot(acc, label=name+' acc; Max acc: ' + str(np.max(acc))[:6] + ', Epoch: ' + str(np.argmax(acc)) )
    plt.title("Acc vs. epochs")
    plt.legend(loc="lower right")
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig(accDir+"acc_compare.png")
    plt.clf()

    for i, name in enumerate(PCTnames):
        if not name in accDict: continue
        val_acc = accDict[name][1]
        plt.plot(val_acc, label=name+' val_acc; Max val_acc: ' + str(np.max(val_acc))[:6] + ', Epoch: ' + str(np.argmax(val_acc)) )
    plt.title("Val_acc vs. epochs")
    plt.legend(loc="lower right")
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig(accDir+"val_acc_compare.png")


    
    plt.close()    


def plotCM(name, BESpredict, truthData, saveDir):
    print(name)
    # Collapse truth labels into 1D (length N_events) array containing truth index (0-5) for each event 
    truthData = np.argmax(truthData, axis=1) 

    cm = confusion_matrix(truthData, np.argmax(BESpredict, axis=1) )
                           
    cmDir = saveDir + "Confusion_Matrices/"
    # Plot Confusion Matrix, both normalized and not normalized
    tools.plot_confusion_matrix(cm, samples, cmDir, name)
    tools.plot_confusion_matrix(cm, samples, cmDir, name, normalize=True)
    print("")

#==================================================================================
# The ROC Curve Comparator --------------------------------------------------------
#----------------------------------------------------------------------------------
# This function creates comparison plots for the PCT roc curves to the BES --------
#     Variables ROC curve ---------------------------------------------------------
#----------------------------------------------------------------------------------

def rocCurveComparator(PCTFiles, PCTlegendNames, PCTcolors):

    #==================================================================================
    #==================================================================================

    # BEST Neural Network /////////////////////////////////////////////////////////////

    #==================================================================================
    #==================================================================================
    

    #==================================================================================
    # Load Test Data //////////////////////////////////////////////////////////////////
    #==================================================================================
    
    # BES variables network for comparison
    # modelType = "newBEST_longLearn/"
    # suffix = "300Basic_300_Z"
    # maskFile = "newBESTMask_300_Z.txt"
    modelDir  = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/"
    # modelType = "newBEST_maskFix/"
    modelType = "recheck_long/"
    # suffix = "longBasic_ak8"
    suffix = "140Basic300Wbothak8HT400"
    pctSuffix = suffix + '_' + '_'.join(PCTnames)
    saveDir = plotDir + pctSuffix + '/'

    logsDir = saveDir + 'logScales/'
    if not os.path.isdir(logsDir): os.makedirs(logsDir)
    pctLossPlotter(PCTnames, saveDir)
    quit()

    modelDir += modelType + suffix + "/"
    # prefix  = "300Basic_"
    # suffix = "300_Z"
    # Load Mask
    # maskFile = "fixBESTMask_ak8.txt"
    maskFile = "300Wbothak8HT400.txt"
    mask, _ = tools.loadMask(modelDir +  maskFile)

    scale = "newBEST_Basic" 
    # scale =  "newBEST_Qmpxy"
    # scale =   "newBEST_Qall"

    # Load h5 Data, set up truth arrays
    dataDict = tools.loadH5Data(h5Dir, mask, sampleTypes, ["test"], scale) 

    #==================================================================================
    # Load BES neural network /////////////////////////////////////////////////////////
    #==================================================================================

    # modelFile = "BEST_model_300Basic_300_Z.h5"
    # modelFile = "BEST_model_longBasic_ak8.h5"
    modelFile = "BEST_model_140Basic300Wbothak8HT400.h5"
    model_BESonly = load_model(modelDir + modelFile)
    print("Loaded the BES variable only neural network")
    
    # Predict
    BESpredict = model_BESonly.predict([dataDict["testEvents"]])
    print("Made predictions using the neural network")

    BEStruth = dataDict["testTruth"]
    with h5py.File("BEST_model_140Basic300Wbothak8HT400_predict.h5", "w") as f:
        f.create_dataset("BESpredict", data=BESpredict)
        f.create_dataset("BEStruth",   data=BEStruth )

    # print("Loading BES predictions...")
    # predictFile = "BEST_model_longBasic_ak8_predict.h5"
    # h5File = h5py.File(modelDir+predictFile, 'r')
    # BESpredict = h5File["BESpredict"][()]
    # BEStruth   = h5File["BEStruth"][()]

    plotCM(suffix, BESpredict, BEStruth, saveDir)

    #==================================================================================
    # Create BES ROC Curves ///////////////////////////////////////////////////////////
    #==================================================================================   
    # compute ROC curve and area for each class
    samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]
    n_classes = len(samples) 
    fprBES = dict()
    tprBES = dict()
    roc_auc_BES = dict()
    for i, sample in enumerate(samples):
        # fprBES[sample], tprBES[sample], _ = roc_curve(dataDict["testTruth"][:, i], BESpredict[:, i])
        fprBES[sample], tprBES[sample], _ = roc_curve(BEStruth[:, i], BESpredict[:, i])
        roc_auc_BES[sample] = auc(fprBES[sample], tprBES[sample])

    # compute micro-average ROC curve and ROC area
    # fprBES["micro"], tprBES["micro"], _ = roc_curve(dataDict["testTruth"].ravel(), BESpredict.ravel() )
    fprBES["micro"], tprBES["micro"], _ = roc_curve(BEStruth.ravel(), BESpredict.ravel() )
    roc_auc_BES["micro"] = auc(fprBES["micro"], tprBES["micro"] )

    # compute macro-average ROC curve and ROC area
    # first aggregate all false positive rates
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


    #==================================================================================
    #==================================================================================

    # Point Cloud Transformer (PCT) ///////////////////////////////////////////////////

    #==================================================================================
    #==================================================================================


    #==================================================================================
    # Load the BEST PCT Data and Model ////////////////////////////////////////////////
    #==================================================================================

    # Load PCT Model
    # if "C" in PCTnames: samples = ["W", "Higgs"]
    # else:               samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]

    # PCTpredictList = []
    # PCTtruthList   = []
    # for i, file_PCT in enumerate(PCTFiles): 
    #     n_classes = len(samples)
    #     if "C" in PCTnames[i]: n_classes = 2

    #     PCTpredictList.append(np.array(file_PCT['DNN']) )
    #     PCTtruthList.append(np.array(to_categorical(file_PCT['pid'], num_classes=n_classes) ) )
    
    # compute ROC curve and area for each class

    print("Loading PCT predictions...")
    pctDict = dict()

    for j, name in enumerate(PCTlegendNames):
        print(name)
        if "C" in PCTnames[j]: samples = ["W", "Higgs"]
        else:                  samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]        
        n_classes = len(samples)
        # print(PCTnames[j], samples)

        PCTfile = PCTFiles[j]
        # print(PCTfile)
        # for key in PCTfile.keys():
            # print(key, PCTfile[key].shape, PCTfile[key][()])
        # print(PCTfile["pid"][380000])
        # print(np.sum(np.array(PCTfile["pid"])))
        PCTpredict = np.array(PCTfile['DNN'])
        PCTtruth   = np.array(to_categorical(PCTfile['pid'], num_classes=n_classes))
        plotCM(name, PCTpredict, PCTtruth, saveDir)
        # print(PCTpredict.shape)
        # print(PCTtruth.shape)
        # print(PCTpredict.shape, PCTpredict)
        # print("")
        # print(PCTtruth.shape, PCTtruth[380000])
        pctDict[name] = dict()
        
        fprPCT = dict()
        tprPCT = dict()
        roc_auc_PCT = dict()

        # PCTtruth = PCTtruthList[j]
        # PCTpredict = PCTpredictList[j]
        for i, sample in enumerate(samples):
            # print(i,sample)
            fprPCT[sample], tprPCT[sample], _ = roc_curve(PCTtruth[:, i], PCTpredict[:, i]) # returns 3 outputs but only care about 2
            roc_auc_PCT[sample] = auc(fprPCT[sample], tprPCT[sample])

        # compute micro-average ROC curve and ROC area
        fprPCT["micro"], tprPCT["micro"], _ = roc_curve(PCTtruth.ravel(), PCTpredict.ravel() )
        roc_auc_PCT["micro"] = auc(fprPCT["micro"], tprPCT["micro"] )

        # compute macro-average ROC curve and ROC area
        # first aggregate all false positive rates
        all_fprPCT = np.unique(np.concatenate([fprBES[sample] for sample in samples]) )

        # interpolate all roc curves
        mean_tprPCT = np.zeros_like(all_fprPCT)
        for sample in samples:
            mean_tprPCT += interp(all_fprPCT, fprPCT[sample], tprPCT[sample] )

        # average and compute macro AUC
        mean_tprPCT /= n_classes

        fprPCT["macro"] = all_fprPCT
        tprPCT["macro"] = mean_tprPCT
        roc_auc_PCT["macro"] = auc(fprPCT["macro"], tprPCT["macro"])

        pctDict[name]["fpr"] = fprPCT
        pctDict[name]["tpr"] = tprPCT
        pctDict[name]["auc"] = roc_auc_PCT
    # quit()
    # print(pctDict)
    #==================================================================================
    # Record ROC AUC //////////////////////////////////////////////////////////////////
    #==================================================================================
    samples = ["W", "Z", "Higgs", "Top", "Bottom", "QCD"]
    print("Recording ROC AUC")


    # Record ROC AUC
    rocLog = open("Logs/PCTcompare_rocLog.txt", "a") 

    rocLog.write("-----------------------------------\n# ")
    rocLog.write(pctSuffix + ":\n")

    # Record BES NN ROC AUC:
    spaces = " "*(30 - len(suffix))
    rocLog.write(suffix + ":" + spaces + "Avg: " + str(roc_auc_BES["macro"])[:8] + ", ")
    for sample in samples: 
        rocLog.write(sample + ": " + str(roc_auc_BES[sample])[:8] + ", ")
    rocLog.write("avg: " + str(roc_auc_BES["micro"])[:8] + "\n")

    # Record PCT ROC AUC:
    for name, thisPCTdict in pctDict.items():
        spaces = " "*(30 - len(name))
        pct_auc = thisPCTdict["auc"]
        rocLog.write(name + ":" + spaces + "Avg: " + str(pct_auc["macro"])[:8] + ", ")
        for sample in samples: 
            if sample not in pct_auc: 
                print("For " + name + ", skipping " + sample )
                continue
            rocLog.write(sample + ": " + str(pct_auc[sample])[:8] + ", ")
        rocLog.write("avg: " + str(pct_auc["micro"])[:8] + "\n")

    rocLog.write("-----------------------------------\n")
    rocLog.close


    #==================================================================================
    # Plot ROC curves /////////////////////////////////////////////////////////////////
    #==================================================================================

    # Plot Loss curves
    # if not os.path.isdir(saveDir): os.makedirs(saveDir)
    logsDir = saveDir + 'logScales/'
    if not os.path.isdir(logsDir): os.makedirs(logsDir)
    pctLossPlotter(PCTnames, saveDir)

    # Plot ROC Curves
    print("Plotting ROC curves")

    # Fill dictionary with plot label information
    labelDict = {} # { key: [plot label, plot title, plot path name], ... }
    for key in roc_auc_BES.keys():
        if   key == "micro": labelDict[key] = ["Micro Average", "average_micro"]
        elif key == "macro": labelDict[key] = ["Macro Average", "average_macro"]
        else:                labelDict[key] = [key + " Category", key]

    for key, rocAUC in roc_auc_BES.items():

        # Assign these for readability:
        # path  = saveDir + labelDict[key][1] + '_ROCplot'
        path  = labelDict[key][1] + '_ROCplot'
        
        # title = pctSuffix + " " + labelDict[key][0] + " ROC Curve" 
        title =  suffix + '\n' + ', '.join(PCTnames) + "\n" + labelDict[key][0] + " ROC Curve" 

        # plt.figure(1)
        plt.figure(figsize=(12.8,9.6))
        # plt.figure(figsize=(12.8,9.6), dpi=100)
        plt.rcParams.update({'font.size': 18})
        plt.plot(fprBES[key], tprBES[key],
                # label= 'BES only ROC Curve (area = {0:0.2f})' ''.format(rocAUC),
                label= 'BES (area = ' + str(rocAUC)[:6] + ') ',
                color='orange', linewidth=2)
        for i, name in enumerate(PCTlegendNames):
            if key not in pctDict[name]["auc"]:
                print("For " + name + ", skipping " + key)
                continue
            fprpct = pctDict[name]["fpr"][key]
            tprpct = pctDict[name]["tpr"][key]
            aucpct = pctDict[name]["auc"][key]
            plt.plot(fprpct, tprpct,
                    label= name +' (area = ' + str(aucpct)[:6] +') ',
                    # color=PCTcolors[i], linewidth=2)
                    linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xticks(np.linspace(0,1,11))
        plt.yticks(np.linspace(0,1,11))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.xscale('log')
        plt.title( title + "_xLogScale")
        plt.legend(loc="upper left")
        plt.show()
        plt.savefig(logsDir + path + "_xlog.png")
        
        plt.yscale('log')
        plt.title( title + "_xyLogScale")
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(logsDir + path + "_xylog.png")

        plt.xscale('linear')
        plt.title( title + "_yLogScale")
        plt.show()
        plt.savefig(logsDir + path + "_ylog.png")

        plt.yscale('linear')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(title)
        plt.savefig(saveDir + path + '.png')
        plt.close()



    print("Check out completed plots at:\n" + saveDir)

# Run the ROC curve plot maker
PCTFiles = [h5py.File(PCTDir + 'h5/' + f + '.h5', 'r') for f in PCTnames ]
# PCTFiles = [h5py.File(PCTDir + 'h5/22/' + f + '.h5', 'r') for f in PCTnames ]
rocCurveComparator(PCTFiles, PCTlegendNames, PCTcolors)

# Plot Loss curves
# saveDir = plotDir + '_'.join(PCTnames) + '/'
# if not os.path.isdir(saveDir): os.makedirs(saveDir)
# saveDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/plots/PCTcompare/300Basic_300_Z_100kS_L_E25_100kS_LV_E31_100kS_LVH_E25"
# pctLossPlotter(PCTnames, saveDir)


# suffix = "300Basic_300_Z"
# saveDir = plotDir + suffix + '_' + '_'.join(PCTnames) + '/'
# pctLossPlotter(PCTnames, saveDir)

