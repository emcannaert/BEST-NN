# modules
import ROOT as root
import numpy as np
import matplotlib
matplotlib.use('Agg') #prevents opening displays (fast), must use before pyplot
import matplotlib.pyplot as plt
import h5py
import argparse, os

################################## NOTES TO SELF ##################################
# Plots need titles, code needs more comments.
# Might be made obsolete by training/plotBESTInputs.py.

# User definitons
# bins_list = [i*100 for i in range(0,40)]
bins_list = [i*50 for i in range(10,40)]

#cmslpc127
# Global variables
years = ["2016_APV","2016","2017","2018"]
# sampleTypes = ["BB","HH","QCD","TT","WW","ZZ"]
sampleTypes = ["BB","HH","QCD","TT","WW","ZZ", "RSG"]
# listOfFileTypes = [".h5","_train.h5","_validation.h5","_test.h5","_train_flattened.h5","_validation_flattened.h5","_test_flattened.h5"]
# listOfFileTypes = ["_train.h5","_validation.h5","_test.h5","_train_flattened.h5","_validation_flattened.h5","_test_flattened.h5"]
# listOfFileTypes = ["_train_flattened.h5"]
listOfFileTypes = [".h5"]

# Main function should take in arguments and call the functions you want
if __name__ == "__main__":
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-s', '--samples',
                        dest='samples',
                        help='Which (comma separated) samples to process. Examples: 1) --all; 2) W,Z,b',
                        default="all")
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        help='Location of directory containing h5 files to plot',
                        default="/uscms/home/bonillaj/nobackup/h5samples_OR/")
                        # default="/uscms/home/bonillaj/nobackup/h5samples_ULv1/")
    parser.add_argument('-y', '--years',
                        dest='years',
                        help='<Required> Which (comma separated) years to process. Examples: 1) all; 2) 2016,2017',
                        # required=True)             
                        default="all")             
    parser.add_argument('-pt', '--ptIndex',
                        dest='ptIndex',
                        type=int,
                        default=142)
                        # default=548)                                   
    parser.add_argument('-ft','--fileTypes',
                        dest='fileTypes',
                        help='Which (comma separated) samples to process. Examples: 1) --all; 2) _train,_test',
                        default="all")
    args = parser.parse_args()
    if not args.samples == "all": sampleTypes = args.samples.split(',')
    if not args.fileTypes == "all": listOfFileTypes = args.fileTypes.split(',')
    if not args.years == "all": years = args.years.split(',')
    plotDir = "plots/"
    if not os.path.isdir(plotDir): os.mkdir(plotDir)
    print("Samples to process: ", sampleTypes)
    print("File types to process: ", listOfFileTypes)

    # Make directories you need
    if not os.path.isdir(args.h5Dir):
        print(args.h5Dir, "does not exist")
        quit()
    
    ## First plot all pt for each collection
    ## So full samples, then train,validation,test, then train_flattened,validation_flattened,test_flattened
    for year in years:
        print("Plotting year", year)
        for suffix in listOfFileTypes:
            print("Plotting suffix", suffix)
            myPtArrays = []
            for sampleType in sampleTypes:
                inputPath = args.h5Dir+sampleType+"Sample_"+year+"_BESTinputs"+suffix
                if sampleType == "RSG": inputPath = args.h5Dir+"TT_ext_Sample_"+year+"_BESTinputs"+suffix
                inputFile = h5py.File(inputPath,"r")
                myPtArrays.append(np.array(inputFile["BES_vars"][...,args.ptIndex]))

            # --- Create histogram, legend and title ---
            plt.figure()
            if suffix == ".h5":
                H = plt.hist(myPtArrays, bins = bins_list, histtype='step', log=True, label=sampleTypes, stacked=False, fill=False, normed=False)
                plt.ylim(top=10000000000)  # adjust the top leaving bottom unchanged
                plt.ylim(bottom=0.1)  # adjust the bottom leaving top unchanged
            else:
                H = plt.hist(myPtArrays, histtype='step', stacked=False, fill=False, bins = bins_list, label=sampleTypes, normed=False)
            leg = plt.legend(frameon=False)
            plt.show()
            savePath = os.path.join(plotDir, "PtDistribution_"+year+"_"+suffix.split('.')[0]+'.png')
            plt.savefig(savePath)
            plt.clf()
            # --- Normalized Create histogram, legend and title ---
            plt.figure()
            H = plt.hist(myPtArrays, histtype='step', stacked=False, fill=False, bins = bins_list, label=sampleTypes, normed=True)
            leg = plt.legend(frameon=False)
            plt.show()
            savePath = os.path.join(plotDir, "PtDistribution_"+year+"_"+suffix.split('.')[0]+'_Normalized.png')
            plt.savefig(savePath)
            plt.clf()

    
    print("Done")

