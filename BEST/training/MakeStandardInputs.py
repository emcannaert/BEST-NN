#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MakeStandardInputs.py ///////////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Reyer Band, Johan S. Bonilla, Brendan Regnary, Mark Samuel Abbott ////
# This program makes Standardized Inputs //////////////////////////////////////////
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Add more comments, improve explanation at the top.
# Save model using joblib instead of saving the mean/variance.
# Make consistent with other scripts.
# Figure out what the issue with scaling is
#       Test by scaling and unscaling in the same script, then plotting. 

import numpy
import h5py
from sklearn import preprocessing
import argparse, os
import tools.functions as tools
listOfSamples = ["Zt","Ht","Wb","QCD"]
setTypes = ["","train","validation","test"]

#==================================================================================
# Standardize BES Vars ////////////////////////////////////////////////////////////
#==================================================================================
def standardizeBESTVars(fileDir = "/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/formatConverter/h5samplesSplit/", sampleTypes = ["Zt","Ht","Wb","QCD","TTBar"], setTypes = ["train,test,validation"], suffix = "flattened", year = "2018"):
    # put BES variables in data frames
    for mySet in setTypes:
        jetBESDF = {}
        besChunks = []
        for mySample in sampleTypes:
            print("Getting", mySample, mySet)
            filePath = fileDir+mySample+"Sample_"+year+"_BESTinputs"
            if not mySet == "":
                filePath = filePath + "_" + mySet
            if not suffix == "":
                filePath = filePath + "_" + suffix
            filePath = filePath + ".h5"
            print("The file path is", filePath)
            myF = h5py.File(filePath,"r")
            jetBESDF[mySample] = myF['BES_vars'][()]
            besChunks = myF['BES_vars'].chunks
            print(type(jetBESDF[mySample]), jetBESDF[mySample].shape)
            myF.close()
            print("Got", mySample, mySet)
        print("Accessed BES variables for", mySet)

        allBESinputs = numpy.concatenate([jetBESDF[mySample] for mySample in sampleTypes])
        print("Shape allBESinputs", allBESinputs.shape)
        if mySet == "train":
            maskPath = "/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/formatConverter/masks/BESvarList.txt"
            _,varDict = tools.loadMask(maskPath)
            #scaler = preprocessing.StandardScaler().fit(allBESinputs)
            scaler = preprocessing.MaxAbsScaler().fit(allBESinputs)
            with open('ScalerParameters_maxAbs_'+mySet+'.txt', 'w') as outputFile:
                for i,iterator in enumerate(scaler.scale_):
                    if not str(i) in varDict:
                        continue
                    outputFile.write('{},{},{},{}\n'.format(varDict[str(i)], "MaxAbs",iterator, 0))
                #for mean,var in zip(scaler.mean_, scaler.var_):        
        print("JetBESDF", jetBESDF.keys())
        for mySample in sampleTypes:
            jetBESDF[mySample] = scaler.transform(jetBESDF[mySample])
            print("Transformed", mySample)
            outFilePath = fileDir+mySample+"Sample_"+year+"_BESTinputs"
            if not mySet == "":
                outFilePath = outFilePath + "_" + mySet
            if not suffix == "":
                outFilePath = outFilePath + "_" + suffix
            outFilePath = outFilePath + "_standardized_maxAbs.h5"
            outF = h5py.File(outFilePath, "w")
            print("Creating Standarized Dataset for ", mySample, len(jetBESDF[mySample]))
            outF.create_dataset('BES_vars', data=jetBESDF[mySample], chunks=(besChunks[0], besChunks[1]), compression='lzf', shuffle=True)
            inFilePath = fileDir+mySample+"Sample_"+year+"_BESTinputs"
            if not mySet == "":
                inFilePath = inFilePath + "_" + mySet
            if not suffix == "":
                inFilePath = inFilePath + "_" + suffix
            inFilePath = inFilePath + ".h5"
            inF = h5py.File(inFilePath, "r")
            # Copy the other datasets (could replace this with soft link?)
            for key in inF.keys():
                if key == "BES_vars": continue
                print("Copying", key)
                inF.copy(inF[key],outF,key)
            inF.close()
            outF.close()
            print("Done creating", outFilePath)
        print("Finished making datasets for", mySet)

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='<Required> Which (comma separated) samples to process. Examples: 1) --all; 2) WW,ZZ,BB',
                        required=True)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="/uscms/home/cannaert/nobackup/BEST/CMSSW_10_6_27/src/BEST/formatConverter/h5samplesSplit/")
    parser.add_argument('-sf','--suffix',
                        dest='suffix',
                        default="flattened")
    parser.add_argument('-y','--year',
                        dest='year',
                        default="2018")
    parser.add_argument('-st','--setType',
                        dest='setType',
                        help='<Required> Which (comma separated) sets to process. Examples: 1) all; 2) train,validation,test',
                        required=True)
    args = parser.parse_args()
    if not args.samples == "all": listOfSamples = args.samples.split(',')
    if not args.setType == "all": setTypes = args.setType.split(',')

    # Make directories you need
    if not os.path.isdir(args.h5Dir): print(args.h5Dir, "does not exist")
    standardizeBESTVars(args.h5Dir, listOfSamples, setTypes, args.suffix, args.year)
    
    print("Done")

