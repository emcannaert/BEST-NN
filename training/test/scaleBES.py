#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MakeStandardInputs.py ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Reyer Band, Johan S. Bonilla, Brendan Regnary, Sam Abbott ////////////
# This program makes Standardized Inputs //////////////////////////////////////////
#----------------------------------------------------------------------------------

import time
startTime = time.time() # Tracks how long script takes

################################## NOTES TO SELF ##################################
# Add more comments, improve explanation at the top.
# Save model using joblib instead of saving the mean/variance.
# Make consistent with other scripts.
# Figure out what the issue with scaling is
#       Test by scaling and unscaling in the same script, then plotting. 

import numpy as np
import h5py
from sklearn import preprocessing
import argparse, os
from sklearn.externals.joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer



sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]

# load data. scale. save scale. repeat.

#==================================================================================
# Standardize BES Vars ////////////////////////////////////////////////////////////
#==================================================================================
def standardizeBESTVars(h5Dir = "../formatConverter/h5samples/", sampleTypes = ["QCD","HH","TT","WW","ZZ","BB"], suffix = "flattened", year = "2017"):
    
    maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask.txt"
    maskFile = open(maskPath, "r")
    maskIndex = []
    for line in maskFile:
      maskIndex.append(line.split(':')[0])
    maskFile.close()
    print(maskPath + " chosen; mask size " + str(len(maskIndex)))
    myMask = [True if str(i) in maskIndex else False for i in range(596)]


# >>> from sklearn.compose import ColumnTransformer
# >>> from sklearn.preprocessing import Normalizer
# >>> ct = ColumnTransformer(
# ...     [("norm1", Normalizer(norm='l1'), [0, 1]),
# ...      ("norm2", Normalizer(norm='l1'), slice(2, 4))])
# >>> X = np.array([[0., 1., 2., 2.],
# ...               [1., 1., 0., 1.]])
# >>> # Normalizer scales each row of X to unit norm. A separate scaling
# >>> # is applied for the two first and two last elements of each
# >>> # row independently.
# >>> ct.fit_transform(X)    
# array([[0. , 1. , 0.5, 0.5],
#        [0.5, 0.5, 0. , 1. ]])
    transformers = () #list of tuples (name, transformer, column(s))

    #       jet _energy scale to 0,1. minmax, or robust/quant
    #       jet_pxpy scale -1,1, maxabs, standard(center on/off)?  robust?     
    #       jet pz needs testing. robust/quantile, standardd, minmax
    #       eta use gaussian scaler. minmax or standard, look at output from both (plot)
    #       check with and without phi later
    #       ak8pt scale to 0,1, minmax
    #       njets/nSecVert, robust?quantile? maxabs?
    #        
    # 
    #  
    #  


    ct = ColumnTransformer(
        [], #transformer list
        remainder = "passthrough"
        #n_jobs = 
    )


    # put BES variables in data frames
    print("Loading h5 files...")
    mySet = "test"
    arrays  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"]) for mySample in sampleTypes]
    # myTrainEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened.h5", "r")["BES_vars"])[:,myMask] for mySample in sampleTypes]
    # myTrainEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_"+mySet+"_"+suffix+".h5", "r")["BES_vars"])[:,myMask] for mySample in sampleTypes]

    print("Accessed all BES variables")

    allBESinputs = np.concatenate(myTrainEvents)
    del myTrainEvents
    print("Shape allBESinputs", allBESinputs.shape)
    
    # print("Fitting Standard Scaler....")
    # scaler = preprocessing.StandardScaler().fit(allBESinputs)    
    # dump(scaler, 'ScalerParameters/ScalerParameters_StandardScaler.joblib')
    # del scaler
    
    # print("Fitting MinMax Scaler....")
    # scaler = preprocessing.MinMaxScaler().fit(allBESinputs)    
    # dump(scaler, 'ScalerParameters/ScalerParameters_MinMax01Scaler.joblib')
    # del scaler

    # print("Fitting MaxAbs Scaler....")
    # scaler = preprocessing.MaxAbsScaler().fit(allBESinputs)    
    # dump(scaler, 'ScalerParameters/ScalerParameters_MaxAbsScaler.joblib')
    # del scaler
    
    # print("Fitting Robust Scaler....")
    # scaler = preprocessing.RobustScaler().fit(allBESinputs)    
    # dump(scaler, 'ScalerParameters/ScalerParameters_RobustScaler.joblib')
    # del scaler
    
    # print("Fitting Quantile Transformer....")
    # scaler = preprocessing.QuantileTransformer(output_distribution='normal').fit(allBESinputs)    
    # dump(scaler, 'ScalerParameters/ScalerParameters_QuantileTransformerNormal.joblib')
    # del scaler
    

    print("Fitting Power Transformer....")
    # preprocessor = make_pipeline(
    #     preprocessing.StandardScaler(with_std=False),
    #     preprocessing.PowerTransformer(standardize=True),
    # )    
    # scaler = preprocessor.fit(allBESinputs)    
    scaler = preprocessing.PowerTransformer().fit(allBESinputs)    
    dump(scaler, 'ScalerParameters/ScalerParameters_power.joblib')
    del scaler


# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    # parser.add_argument('-s', '--samples',
    #                     dest='samples',
    #                     help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) WW,ZZ,BB',
    #                     required=True)
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
    # if not args.samples == "all": sampleTypes = args.samples.split(',')

    # Make directories you need
    if not os.path.isdir(args.h5Dir): print(args.h5Dir, "does not exist")
    if not os.path.isdir("ScalerParameters"): os.makedirs("ScalerParameters")
    standardizeBESTVars(args.h5Dir, sampleTypes, args.suffix, args.year)
    

    # Check how long the script took to run
    timelog = open("Logs/Scaler_TimeLog.txt", "a") 
    timeTaken = divmod(time.time() - startTime, 60.)
    timeMessage = "Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
    print(timeMessage)
    timelog.write(timeMessage)
    timelog.close

