import time
startTime = time.time() # Tracks how long script takes

import numpy as np
import h5py
from sklearn import preprocessing
import argparse, os
from sklearn.externals.joblib import dump, load
from sklearn.pipeline import make_pipeline



sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
setTypes = ["validation","test","train"]
# load data. scale. save scale. repeat.

#==================================================================================
# Standardize BES Vars ////////////////////////////////////////////////////////////
#==================================================================================
def standardizeBESTVars(h5Dir, sampleTypes, setTypes, suffix, year):
    
    # maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/oldBESTMask.txt"
    maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/newBESTMask_allFrames.txt"
    print("Loading mask: " + maskPath)
    maskFile = open(maskPath, "r")
    maskIndex = []
    for line in maskFile:
      maskIndex.append(line.split(':')[0])
    maskFile.close()
    print("Mask size: " + str(len(maskIndex)))
    myMask = [True if str(i) in maskIndex else False for i in range(596)]


    # # put BES variables in data frames
    # print("Loading h5 files...")
    # myTrainEvents = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_train_flattened.h5", "r")["BES_vars"])[:,myMask] for mySample in sampleTypes]

    # print("Accessed all BES variables")

    # allBESinputs = np.concatenate(myTrainEvents)
    # del myTrainEvents
    # print("Shape allBESinputs", allBESinputs.shape)
    
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
    
    scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/"
    if not os.path.isdir(scaleDir): 
        print(scaleDir, "does not exist")
        quit()
    scales =  [ "newBEST_Basic", "newBEST_Qmpxy", "newBEST_Qall" ]

    # besChunks = h5py.File(h5Dir+"QCDSample_2017_BESTinputs_test_flattened.h5","r")["BES_vars"].chunks
    besChunks = [10, 551]
    # put BES variables in data frames
    for mySet in setTypes:
        print(mySet)
        preScaleEvents  = [np.array(h5py.File(h5Dir+mySample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")["BES_vars"])[:,myMask] for mySample in sampleTypes]

        print("Accessed BES variables for", mySet)
        print("Pre scale events shape:", [arr.shape for arr in preScaleEvents])

        # allBESinputs = np.concatenate([jetBESDF[mySample] for mySample in sampleTypes])
        # print("Shape allBESinputs", allBESinputs.shape)
        # scaler = preprocessing.StandardScaler().fit(allBESinputs)

        # with open('ScalerParameters_'+mySet+'.txt', 'w') as outputFile:
        #     for mean,var in zip(scaler.mean_, scaler.var_):
        #         outputFile.write('{},{}\n'.format(mean, var))

        for scale in scales:

            scalePath = scaleDir + scale + ".joblib"
            if not os.path.isfile(scalePath):
                print(scalePath, "does not exist")
                quit()
            scaler = load(scalePath)
            print(scale, scaler.get_params())

            for i, arr in enumerate(preScaleEvents):
                mySample = sampleTypes[i]
                scaledData = scaler.transform(arr)
                print("Transformed", mySample)
                outFilePath = h5Dir+mySample+"Sample_"+year+"_BESTinputs"
                if not mySet == "":
                    outFilePath = outFilePath + "_" + mySet
                if not suffix == "":
                    outFilePath = outFilePath + "_" + suffix
                outFilePath = outFilePath + "_" + scale + ".h5"
                outF = h5py.File(outFilePath, "w")
                print("Creating Standarized Dataset for ", mySample, len(scaledData))
                outF.create_dataset('BES_vars', data=scaledData, chunks=(besChunks[0], besChunks[1]), compression='lzf', shuffle=True)
                outF.close()
                del scaledData
                print("Done creating", outFilePath)
            print("Finished making datasets for", scale)
    print("Finished making datasets for", mySet)


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
    # scaleDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters"
    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    # if not os.path.isdir(scaleDir): 
        # print(scaleDir, "does not exist")
        # quit()

    standardizeBESTVars(args.h5Dir, sampleTypes, setTypes, args.suffix, args.year)
    

    # Check how long the script took to run
    # timelog = open("Logs/Scaler_TimeLog.txt", "a") 
    timeTaken = divmod(time.time() - startTime, 60.)
    timeMessage = "Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.\n"
    print(timeMessage)
    # timelog.write(timeMessage)
    # timelog.close

