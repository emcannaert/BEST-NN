#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# mask.py /////////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Samantha Abbott ------------------------------------------------------
# This script generates masks to be used in the training step. ////////////////////
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Figure out how this script should fit into the final release (will we keep it?).
# Does this belong in formatConverter or training?
# If we keep it, it needs to be improved and needs more comments.

import os
samples = ['W', 'Z', 'H', 'Top', 'b', 'QCD']
import numpy as np
from sklearn.externals.joblib import load

# varFile = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/h5samples/BESvarList.txt"
# Fixed for Basic scale h5 file:
# varFile = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/old/fix/allBESTVars.txt"
logDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/Logs/"
# inFile = logDir + "recheck_classifyLog.txt"
inFile = logDir + "recheck_long_2_classifyLog.txt"
nameDict = {}
classDict = {}
models = []
with open(inFile, 'r') as f:
    for i, line in enumerate(f):
        # if i < 430: continue
        # if i < 650: continue
        # if i < 670: continue
        # if i < 710: continue
        # if i < 1020: continue
        # if i < 1190: continue
        # if i < 1240: continue
        # if i < 1320: continue
        # if i < 1350: continue
        if "-" in line: continue
        if "Total" in line: continue
        if "Running " in line:
            name = line.strip()[8:-1]
            node = name[:3]
            frame = name[8:]
            model = node + '_' + frame
            
            nameDict[model] = name
            classDict[model] = {}
            models.append(model)            
            # f.next()
        else:
            thisline = line.strip()
            sample = thisline.split(' ')[0]
            ratio = thisline[-22:-15]
            classDict[model][sample] = ratio
models.sort()

keys = ["acc", "val_acc", "loss", "val_loss"]

# for model, mydict in classDict.items(): print(model, mydict)
outFile = logDir + "CLASSLOG.txt"
with open(outFile, 'a') as f:
    f.write('Categories:\t\t' + str(samples) + '\n')
    for model in models:
        f.write(model + ": \t\t")
        for sample in samples:
            f.write(classDict[model][sample] + ', ')
        f.write('\n')


# modelsDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/recheck/"
modelsDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/recheck_long_2/"
outFile = os.path.join(logDir,"ACCLOG.txt")
with open(outFile, 'a') as f:
    f.write('Metrics: \t\t ' + str(keys) + '\n')
    for model in models:
        name = nameDict[model]
        if "140Basic300Wak8SDHT" in name: continue
        historyPath = os.path.join(modelsDir,name,"history_"+name+".joblib")
        history = load(historyPath)
        accIndex = np.argmax(history['val_acc'])
        
        # Record max accuracy
        f.write(model + ": \t\t")
        for key in keys:
            f.write(str(history[key][accIndex])[:5] + ', ')
        f.write('\n')