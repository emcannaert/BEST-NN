#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MakeStandardInputs.py ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Sam Abbott ///////////////////////////////////////////////////////////
# This program Standardizes the BEST Inputs ///////////////////////////////////////
#----------------------------------------------------------------------------------
import tools.functions as tools
startTime = tools.logTime() # Tracks how long script takes

import numpy as np
import h5py
import argparse, os
from sklearn.externals.joblib import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer

# sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
sampleTypes = ["WW","ZZ","HH","ZPTT","BB","QCD"]

# It is important that "train" is FIRST in this list!!!!
setTypes = ["train", "validation", "test"]
    
def checkRepeat(j, keys):
    repeatList = []
    while j in keys:
        repeatList.append(j)
        j += 1
    # print(repeatList)
    return repeatList

def standardizeBESTVars(h5Dir, scaleDir, maskPath, sampleTypes, suffix, year):  
    #==================================================================================
    # Prepare Scaler //////////////////////////////////////////////////////////////////
    #==================================================================================
    # num_BES_inputs = h5py.File(h5Dir+sampleTypes[0]+"Sample_"+year+"_BESTinputs_train"+suffix+".h5","r")["BES_vars"].shape[1]

    # maskPath = "../formatConverter/masks/BESTMask.txt"
    # maskPath = "../formatConverter/h5samples/BESvarList.txt"
    vars = []
    inds = []
    scalerDict = { "standard":{}, "minmax":{}, "maxabs":{}, "noscale":{} }
    
    # Load in the desired mask, and sort the events to be scaled accordingly.
    with open(maskPath, "r") as maskFile:
        for i, line in enumerate(maskFile):
            index, var = line.split(':')
            var = var.strip()

            inds.append(index)
            vars.append(var)

            if "jetAK8_pt" in var: # Scale pT from [500,2000] to [0,1]
                scalerDict["minmax"][i] = (var) 
            elif "jet_p" in var: # Scale px,py,pz to 0 mean and unit variance
                scalerDict["standard"][i] = (var)  
            elif "jet_energy" in var: # Normalize energy by dividing by max value, giving [0,1] 
                scalerDict["maxabs"][i] = (var)
            elif "mass" in var.lower(): # Normalize mass by dividing by max value, giving [0,1]
                scalerDict["maxabs"][i] = (var)  
            elif "jetAK8_eta" in var: # Normalize eta by dividing by max value, giving [0,1]
                scalerDict["maxabs"][i] = (var)
            else: # All other variables are not scaled (they are already close to [-1,1] or [0,1])
                scalerDict["noscale"][i] = (var)
    
    # Currently we do not generate unused vars
    # mask = [True if str(i) in inds else False for i in range(num_BES_inputs)]

    # Create list of transformers to apply, in order of when the events appear.
    # Use checkRepeat to see how many events in a row use the same transformer.
    transformers = []
    i = 0
    while i < len(vars):
        var = vars[i]
        for scaleKey, indexDict in scalerDict.items(): #scaleKey,indexDict = "minmax", {index:(var), ....}
            if not (var in indexDict.values()): continue # Only append to transformers the correct scaler for this var 
            varKeys = list( map(int, indexDict.keys()) ) # this is a list of the vars index for this mask for this ct for this scaler, sorted.
            varKeys.sort()
            repeats = checkRepeat(i, varKeys)
            transName = scaleKey + '_' + str(i)

            if scaleKey == "standard":  transformers.append( (transName, StandardScaler(), repeats) )  
            elif scaleKey == "minmax":  transformers.append( (transName, MinMaxScaler(), repeats) )
            elif scaleKey == "maxabs":  transformers.append( (transName, MaxAbsScaler(), repeats) )
            elif scaleKey == "noscale": transformers.append( (transName, "passthrough", repeats) )
        i += len(repeats)
    del i
    
    #==================================================================================
    # Fit Scaler //////////////////////////////////////////////////////////////////////
    #==================================================================================

    # IMPORTANT: ONLY FIT THE SCALE MODEL ON THE TRAINING SET. THEN, APPLY THAT MODEL TO EVERYTHING ELSE.
    print("Loading pre scale h5py files")
    preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_train"+suffix+".h5","r")["BES_vars"])[()] for mySample in sampleTypes]
    # preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_train"+suffix+".h5","r")["BES_vars"])[:,mask] for mySample in sampleTypes]
    print("Pre scale events shape:", [arr.shape for arr in preScaleEvents])

    print("Concatenating...")
    preScaleAll = np.concatenate(preScaleEvents)
    # del preScaleEvents
    print("Pre scale events shape:", preScaleAll.shape)

    ct = ColumnTransformer(
        transformers, #transformer list
        remainder = "drop", # there should not be any remainders
        n_jobs = 10
    )

    # print(ct)
    # print(ct.get_params())
    # print("")
    # for key, val in customScalers[scaler].items(): 
    #     print(key, val)
    #     print("")
    # print(customScalers[scaler])
    # print("")
    # quit()

    print("Fitting...")
    ct.fit(preScaleAll)
    del preScaleAll

    #==================================================================================
    # Record Scaler ///////////////////////////////////////////////////////////////////
    #==================================================================================

    # This form of the scaler is easy to load with python ( sklearn.externals.joblib.load(scalePath) )
    scalePath = os.path.join(scaleDir,'BESTScalerParameters.joblib')
    print("Saving Model: " + scalePath)
    dump(ct, scalePath)

    # This form of the scaler is for manually recreating the scaler, specifically for the NTuplizer C++ code used later
    scalePath = os.path.join(scaleDir,'BESTScalerParameters.txt')
    print("Saving Parameters: " + scalePath)
    with open(scalePath, 'w') as f: # Comments below describe transformation applied
        for name, transformer, events in ct.transformers_: 
            numEvents = len(events)
            # print(name)
            if "noscale" in name: # scaledvar = var
                nameKey = "NoScale"
                param1 = [0]*numEvents
                param2 = [0]*numEvents

            elif "min" in name: # scaledvar = ( var - var_min[param1] ) / ( var_max[param2] - var_min[param1] )
                """ Notes on this transformation:
                var_max = maximum value for this feature in training set ( equivalent to np.max(this_row_of_data) ) 
                var_min = minimum value for this feature in training set ( equivalent to np.min(this_row_of_data) )
                
                Note: The above MinMax transformation is a simplification that is true when the output is [0,1].
                General form: X_std = ( var - var_min[param1] ) / ( var_max[param2] - var_min[param1] )
                                scaledvar = f_min + ( X_std * (f_max-f_min) )
                When [f_min,f_max] are set to the default values of [0,1], the general form simplifies to scaledvar = X_std. 
                
                Another Note: Could use scale_ and min_ rather than data_min_ and data_max_. 
                                In this case, the transformation would be:
                                scaledvar = min[param2] + ( var * scale[param1] )
                This form of the transformation has the benefit of being invariant to the choice of f_min and f_max.
                However, the other form of the transformation was chosen as it was found to be slightly more accurate.
                To record this version of the transformation, simply swap the comments on param1 and param2 below.
                """
                nameKey = "MinMax"
                param1 = transformer.data_min_
                param2 = transformer.data_max_
                # param1 = transformer.scale_
                # param2 = transformer.min_

            elif "standard" in name: # scaledvar = (var - mean[param2]) / std_dev[param1]
                nameKey = "Standard"
                param1 = transformer.scale_
                param2 = transformer.mean_

            elif "abs" in name: # scaledvar = var / max_value[param1]
                nameKey = "MaxAbs"
                param1 = transformer.max_abs_
                # param2 = transformer.scale_
                param2 = [0]*numEvents # scale_ and max_abs_ are the same parameter

            for i, event in enumerate(events):
                f.write('{},{},{},{}\n'.format(event, nameKey, param1[i], param2[i]))

    #==================================================================================
    # Transform and save data /////////////////////////////////////////////////////////
    #==================================================================================

    for mySet in setTypes: # MAKE SURE THAT TRAIN IS FIRST IN THE setTypes LIST!!!!!
        if not mySet == "train":
            preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_"+year+"_BESTinputs_"+mySet+suffix+".h5","r")["BES_vars"])[:,mask] for mySample in sampleTypes]
        for i, arr in enumerate(preScaleEvents):
            print("Transforming ", mySample)
            mySample = sampleTypes[i]
            scaledData = ct.transform(arr)

            print("Creating Standarized Dataset for ", mySample, len(scaledData))
            # We are not keeping the RSG samples, so the ZPrime samples can just be named TTSamples from now on
            if mySample == "ZPTT": outFilePath = h5Dir+"TTSample_"+year+"_BESTinputs_"+mySet+suffix+"_standardized.h5"
            else:                  outFilePath = h5Dir+mySample+"Sample_"+year+"_BESTinputs_"+mySet+suffix+"_standardized.h5"
    
            with h5py.File(outFilePath, "w") as outF:
                # outF.create_dataset('BES_vars', data=scaledData, chunks=(10, num_BES_inputs), compression='lzf', shuffle=True)
                outF.create_dataset('BES_vars', data=scaledData, compression='lzf', shuffle=True)
            del scaledData

            print("Done creating", outFilePath)
        del preScaleEvents
    del ct

if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to standardize data for training.')
    parser.add_argument('-hd','--h5Dir', dest='h5Dir',
                        default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/",
                        help="Input File Dir [default: /uscms/home/bonillaj/nobackup/h5samples_ULv1/]")
    parser.add_argument('-sf','--suffix', dest='suffix',
                        default="flattened",
                        help="Suffix, used to select correct h5 input file to standardize [default: 'flattened']")
    parser.add_argument('-y','--year', dest='year',
                        default=["2016_APV","2016","2017","2018"],
                        help='Year of data taking to use [default: ["2016_APV","2016","2017","2018"]]')
    parser.add_argument('-sd','--scaleDir', dest='scaleDir',
                        default="ScalerParameters",
                        help="Dir to store scale params and scaler object to check later [default: ScalerParameters]")
    parser.add_argument('-mp','--maskPath', dest='maskPath',
                        default = "../formatConverter/masks/BESTMask.txt",
                        help="Path to mask file [default: ../formatConverter/masks/BESTMask.txt]")
    args = parser.parse_args()

    suffix = args.suffix
    if not suffix == "": suffix = "_" + suffix

    # Check inputs
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    if not os.path.isfile(args.maskPath): 
        print(args.maskPath, "does not exist")
        quit()
    if not os.path.isdir(args.scaleDir): os.makedirs(args.scaleDir)

    for year in args.year:
        standardizeBESTVars(args.h5Dir, args.scaleDir, args.maskPath, sampleTypes, suffix, year)

    tools.logTime(startTime)