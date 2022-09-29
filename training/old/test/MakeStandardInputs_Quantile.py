#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MakeStandardInputs.py ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Sam Abbott ///////////////////////////////////////////////////////////
# This program Standardizes the BEST Inputs ///////////////////////////////////////
#----------------------------------------------------------------------------------
import tools.functions as tools

startTime = tools.logTime() # Tracks how long script takes

################################## NOTES TO SELF ##################################
# This is similar to MakeStandardInputs, but with the option of scaling using the Quantile Transformers

import numpy as np
import h5py
import argparse, os
from sklearn.externals.joblib import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer


sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
setTypes = ["train", "validation", "test"]

    
def checkRepeat(j, keys):
    repeatList = []
    while j in keys:
        repeatList.append(j)
        j += 1
    # print(repeatList)
    return repeatList

def standardizeBESTVars(h5Dir, sampleTypes, suffix, year):  
    #==================================================================================
    # Load Data ///////////////////////////////////////////////////////////////////////
    #==================================================================================
    num_BES_inputs = h5py.File(h5Dir+sampleTypes[0]+"Sample_"+year+"_BESTinputs_train"+suffix+".h5","r")["BES_vars"].shape[1]

    maskPath = "../formatConverter/masks/BESTMask.txt"
    # maskPath = "../formatConverter/h5samples/BESvarList.txt"
    allVars = []
    varDict = {}
    # revDict = {}
    customScalers = { 
                "Basic": { "standard":{}, "minmax":{}, "maxabs":{},              "noscale":{} }, 
                "Qmpxy": { "standard":{}, "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }, 
                "Qall":  {                "minmax":{}, "maxabs":{}, "quantn":{}, "noscale":{} }
              }
    scalers = customScalers.keys()
    scalers.sort()
    print(scalers)
    i = 0
    
    # Load in the desired mask, and sort the events to be scaled accordingly.
    maskFile = open(maskPath, "r")
    for line in maskFile:
        index, var = line.split(':')
        var = var.strip()
        varDict[index] = var
        # revDict[var] = index
        allVars.append(var)

        if "jetAK8_pt" in var:
            for scalerDict in customScalers.values(): scalerDict["minmax"][i] = (var) 
        elif "jet_energy" in var:
            for scaler, scalerDict in customScalers.items():
                if ("Qall" in scaler): scalerDict["quantn"][i] = (var)
                else:                  scalerDict["minmax"][i] = (var)
        elif "jet_pz" in var:
            for scaler, scalerDict in customScalers.items():
                if ("Qall" in scaler): scalerDict["quantn"][i] = (var)
                else:                  scalerDict["standard"][i] = (var)                
        elif "jet_p" in var:
            for scaler, scalerDict in customScalers.items():
                if ("Q" in scaler): scalerDict["quantn"][i] = (var)
                else:               scalerDict["standard"][i] = (var)  
        elif "mass" in var.lower():
            for scaler, scalerDict in customScalers.items():
                if ("Q" in scaler): scalerDict["quantn"][i] = (var)
                else:               scalerDict["maxabs"][i] = (var)  
        elif "jetAK8_eta" in var:        
            for scaler, scalerDict in customScalers.items(): scalerDict["maxabs"][i] = (var)
        else: 
            for scalerDict in customScalers.values(): scalerDict["noscale"][i] = (var)
        i += 1
    maskFile.close()
    del i        
    mask = [True if str(i) in varDict else False for i in range(num_BES_inputs)]

    # For each Custom Scaler, create list of transformers to apply, in order of when the events appear.
    # Use checkRepeat to see how many events in a row use the same transformer.
    transDict = {}
    for scaler in scalers:
        # print(scaler)
        scalerDict = customScalers[scaler]
        transDict[scaler] = []
        i = 0
        while i < len(allVars):
            var = allVars[i]
            # trueIndex = revDict[var]
            # print(i,trueIndex,var)

            for myScale, myDict in scalerDict.items(): #myScale,myTrans = "minmax", {index:(var), ....}
                # print(myScale, myDict)
                # print(myDict.values())
                # ivd = {v: k for k, v in myDict.items()}

                if not (var in myDict.values()): continue
                varKeys = list( map(int, myDict.keys()) ) # this is a list of the allVars index for this mask for this ct for this scaler, sorted.
                varKeys.sort()
                repeats = checkRepeat(i, varKeys)
                transName = scaler + '_' + myScale + '_' + str(i)
                # print(transName, repeats)

                transformers = transDict[scaler]
                if myScale == "standard":  transformers.append( (transName, StandardScaler(), repeats) )  
                elif myScale == "minmax":  transformers.append( (transName, MinMaxScaler(), repeats) )
                elif myScale == "maxabs":  transformers.append( (transName, MaxAbsScaler(), repeats) )
                elif myScale == "quantn":  transformers.append( (transName, QuantileTransformer(n_quantiles = 350000, subsample = 20000000, output_distribution = 'normal'), repeats) )  
                elif myScale == "noscale": transformers.append( (transName, "passthrough", repeats) )
                
                else: 
                    print("bad thing happened")
                    quit()
            i += len(repeats)

    print("Loading pre scale h5py files")
    preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_train"+suffix+".h5","r")["BES_vars"])[:,mask] for mySample in sampleTypes]
    print("Pre scale events shape:", [arr.shape for arr in preScaleEvents])

    print("Concatenating...")
    preScaleAll = np.concatenate(preScaleEvents)
    # del preScaleEvents
    print("Pre scale events shape:", preScaleAll.shape)

    # ==================================================================================
    # Standardize BES Vars ////////////////////////////////////////////////////////////
    # ==================================================================================
    
    print("Starting loop...")    
    for scaler in scalers:
        print(scaler)

        ct = ColumnTransformer(
            transDict[scaler], #transformer list
            remainder = "drop", # there should not be any remainders
            # n_jobs = None
        )

        # print(ct)
        # print(ct.get_params())
        # print("")
        # for key, val in customScalers[scaler].items(): 
        #     print(key, val)
        #     print("")
        # print(customScalers[scaler])
        # print("")

        print("Fitting...")
        ct.fit(preScaleAll)
        del preScaleAll

        scalePath = 'ScalerParameters/' + scaler + '.joblib'
        print("Saving Model: " + scalePath)
        dump(ct, scalePath)

        scalePath = 'ScalerParameters/' + scaler + '.txt'
        print("Saving Parameters: " + scalePath)
        with open(scalePath, 'w') as f:
            for name, transformer, events in ct.transformers_: 
                numEvents = len(events)
                # print(name)
                if "noscale" in name:
                    nameKey = "NoScale"
                    param1 = [0]*numEvents
                    param2 = [0]*numEvents
                else:    
                    if "min" in name:
                        nameKey = "MinMax"
                        param1 = transformer.data_min_
                        param2 = transformer.data_max_
                    elif "standard" in name:
                        nameKey = "Standard"
                        param1 = transformer.scale_
                        param2 = transformer.mean_
                    elif "abs" in name:
                        nameKey = "MaxAbs"
                        param1 = transformer.max_abs_
                        param2 = [0]*numEvents #scale_ and max_abs_ are the same parameter
                for i, event in enumerate(events):
                    f.write('{},{},{},{}\n'.format(event, nameKey, param1[i], param2[i]))

        for mySet in setTypes:
            if not mySet == "train":
                preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_"+mySet+suffix+".h5","r")["BES_vars"])[:,mask] for mySample in sampleTypes]
            for i, arr in enumerate(preScaleEvents):
                print("Transforming ", mySample)
                mySample = sampleTypes[i]
                scaledData = ct.transform(arr)

                print("Creating Standarized Dataset for ", mySample, len(scaledData))
                outFilePath = h5Dir+mySample+"Sample_"+year+"_BESTinputs_"+mySet+suffix+"_"+scaler+".h5"
       
                outF = h5py.File(outFilePath, "w")
                outF.create_dataset('BES_vars', data=scaledData, chunks=(10, num_BES_inputs), compression='lzf', shuffle=True)
                outF.close()
                del scaledData

                print("Done creating", outFilePath)
            del preScaleEvents
        del ct

if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) WW,ZZ,BB',
                        default="all")
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="flattened")
    parser.add_argument('-y','--year',
                        dest='year',
                        default="2017")

    args = parser.parse_args()
    if not args.samples == "all": sampleTypes = args.samples.split(',')
    suffix = args.suffix
    if not suffix == "": suffix = "_" + suffix

    scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters"
    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isdir(scaleDir): 
        print(scaleDir, "does not exist")
        quit()

    standardizeBESTVars(args.h5Dir, sampleTypes, suffix, args.year)

    tools.logTime(startTime)