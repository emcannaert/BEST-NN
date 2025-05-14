#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# sampleFlattener.py /////////////////////////////////////////////////////
#==================================================================================
# Author(s): Johan S Bonilla, Samantha Abbott, Brendan Regnery, Reyer Band --------
# This program takes in h5 files and equalizes the number of events based on pt ///
# Inputs should be h5 files after splitting the samples
# The process plots the various files and, bin by bin, keeps all events of the
# least populous sample, and throws away events based on probabilty ratio
# P=NumEventsMinSample/NumEventsThisSample
# Output are h5 samples.
#----------------------------------------------------------------------------------

import h5py
import argparse
import os
import numpy as np
import numpy.ma as ma
from sklearn.model_selection import train_test_split
import pickle

# years = ["2015","2016","2017","2018"]
years = ["combine"]
# years = ["2017"]
decays_types = ["allDecays"]
#decays_types = ["","allDecays"]

#mass_types = ["low_mass", "high_mass", "all_mass"]
mass_types = ["all_mass"]

# sampleTypes_ = ["HT", "ZT", "WB", "QCD","Top"]
sampleTypes_ = ["HT", "ZT", "WB","bg"]
# setTypes = ["","train","validation","test"]
setTypes = ["train"]



def print_hist2d_matrix(hist, x_edges, y_edges):
    # Print column headers
    print " " * 12,
    for j in range(len(y_edges) - 1):
        print "%7.1f-%-7.1f" % (y_edges[j], y_edges[j+1]),
    print ""

    print "-" * (12 + (len(y_edges) - 1) * 15)

    # Loop over rows (x-axis bins)
    for i in range(len(x_edges) - 1):
        # Row label
        print "%7.1f-%-7.1f |" % (x_edges[i], x_edges[i+1]),
        
        # Loop over columns (y-axis bins)
        for j in range(len(y_edges) - 1):
            count = hist[i][j]
            print "%13.0f" % count,
        print ""

## Loop over all files and keep/reject events in batches. One uses train_test_split to do the heavy lifting.
## The random state in the function ensures the same mask across the keys. Since the probabilities are binned in pt,
## one must loop over all pt bins and evaluate keep/reject for each.
def flattenFile(keepProbs_2D, h5Dir, outDir, sampleTypes, year, setType, bins_SJ, bins_HT, userBatchSize, mass_type, args):
    print("Begin flattening for", setType)
        
    HT_max = args.rangeHighHT
    HT_min = args.rangeLowHT

    SJ_mass_max = args.rangeHighSJMass
    SJ_mass_min = args.rangeLowSJMass

    N_bins_HT = args.nBinsHT # represents HT
    N_bins_SJ = args.nBinsSJMass  # represents SJ mass

    hist_to_count = []

    hist_to_count2 = []
    all_events_to_count2 = []
    probs_to_count2 = []


    what_post_should_be = []

    # Loop over all samples and flatten each one by one
    for sample in sampleTypes:


        hist_2d, _, _ = np.histogram2d([], [], bins=[bins_HT, bins_SJ])
        hist_to_count.append( hist_2d  )


        hist_to_count2.append(np.zeros( (len(bins_HT)-1, len(bins_SJ)-1 ) ))
        all_events_to_count2.append(np.zeros( (len(bins_HT)-1, len(bins_SJ)-1 ) ))
        probs_to_count2.append(np.zeros( (len(bins_HT)-1, len(bins_SJ)-1 ) ))
        what_post_should_be.append(np.zeros( (len(bins_HT)-1, len(bins_SJ)-1 ) ))
        mass_str = mass_type + "_"
        sample_str = sample + "_"

        filePath = h5Dir + sample_str + 'Sample_' + mass_str + year + '_BESTinputs'
        if setType == "":
            filePath = filePath + ".h5"
        else:
            filePath = filePath + "_" + setType + ".h5"

        fIn = h5py.File(filePath, 'r')
        fOut = h5py.File(outDir + filePath.split('.')[-2].split('/')[-1] + "_flattened.h5", "w")
        besData = {}
        counter = 0
        totalEvents = fIn[list(fIn.keys())[0]].shape[0]
        print("Begin batching for sample", sample, "in year", year, "total events", totalEvents)
        


        # Perform the keep/throw-away operation in batches
        while counter < totalEvents:
            batchSize = userBatchSize if (totalEvents > counter + userBatchSize) else (totalEvents - counter)
            
            saved_events_batch_HT = []
            saved_events_batch_SJ_mass = []

            # Grab SJ mass and HT data for the current batch
            sjMassData = np.array(fIn["BES_vars"][counter:counter + batchSize, 88])
            htData = np.array(fIn["BES_vars"][counter:counter + batchSize, 111])

            # Process each key in the dataset
            for myKey in fIn.keys():
                myKeyData = np.array(fIn[myKey][counter:counter + batchSize, ...])
                dsetShape = fIn[myKey].shape
                dsetChunks = fIn[myKey].chunks
                if dsetChunks is None:
                    dsetChunks = (min(1000, dsetShape[0]), dsetShape[1])  # Default chunk size
                
                # Loop over 2D bins (SJ mass and HT)
                for i in range(0,len(bins_SJ)-1):
                    for j in range(0, len(bins_HT)-1):
                        prob = keepProbs_2D[sampleTypes.index(sample)][i][j]
                        if prob == 0:
                            continue
                        
                        # Define the current bin ranges
                        sjMassLow, sjMassHigh = bins_SJ[i], bins_SJ[i + 1]
                        htLow, htHigh = bins_HT[j], bins_HT[j + 1]
                        
                        # Identify events in the current 2D bin
                        sjMassBool = (sjMassLow < sjMassData) & (sjMassData < sjMassHigh)
                        htBool = (htLow < htData) & (htData < htHigh)
                        combinedBool = sjMassBool & htBool
                        
                        result_old = myKeyData[combinedBool]

                        saved_events_batch_HT.extend(sjMassData[combinedBool])
                        saved_events_batch_SJ_mass.extend(htData[combinedBool])


                        if result_old.shape[0] == 0:
                            continue
                        

                        # Apply the probability to keep events
                        if prob < 1:
                            result = train_test_split(result_old, train_size=prob, shuffle=True, random_state=29)[0]
                            #print("For sample %s --------- bin %s/%s (HT=%s, SJ mass = %s), the probability is %s, old bin size / new bin size = %s (original # events= %s, new # events = %s)"%(sample, i,j, htLow, sjMassLow, prob, float(len(result)) / float(len(result_old)), len(result_old), len(result) ))
                        else:
                            result = result_old
                            #print("For sample %s --------- bin %s/%s (HT=%s, SJ mass = %s), the probability is %s, old bin size / new bin size = %s (original # events= %s, new # events = %s)"%(sample, i,j, htLow, sjMassLow, prob, float(len(result)) / float(len(result_old)), len(result_old), len(result) ))

                        #num_counts_in_bin = len(result)

                        if len(result) == 0:
                            continue

                        hist_to_count2[sampleTypes.index(sample)][i][j] += len(result)
                        all_events_to_count2[sampleTypes.index(sample)][i][j] += len(result_old)
                        probs_to_count2[sampleTypes.index(sample)][i][j] = prob
                        what_post_should_be[sampleTypes.index(sample)][i][j] += prob*len(result_old)


                        # Store the kept data
                        if myKey not in besData.keys():
                            if myKey == "BES_vars":
                                besData[myKey] = fOut.create_dataset(
                                    myKey, data=result, maxshape=(None, dsetShape[1]),
                                    chunks=(dsetChunks[0], dsetChunks[1]), compression='lzf', shuffle=True
                                )
                        else:
                            besData[myKey].resize(besData[myKey].shape[0] + len(result), axis=0)
                            besData[myKey][-len(result):] = result
            counter += batchSize
            hist_2d_batch , _, _ = np.histogram2d(saved_events_batch_HT, saved_events_batch_SJ_mass, bins=[bins_HT, bins_SJ])  
            hist_to_count[sampleTypes.index(sample)] += hist_2d_batch.T
        print_hist2d_matrix(hist_to_count[sampleTypes.index(sample)],  bins_HT, bins_SJ)

    print("======== Final pre-flat signal histogram ==========")
    print_matrix(all_events_to_count2[0],0,1)
    print("========    Final pre-flat BR histogram  ==========")
    print_matrix(all_events_to_count2[1],0,1)

    print("======== WHAT POST-FLAT SIGNAL SHOULD BE ==========")
    print_matrix(what_post_should_be[0],0,1)
    print("======== WHAT POST-FLAT BR SHOULD BE ==========")
    print_matrix(what_post_should_be[1],0,1)


    print("======== Final post-flat signal histogram ==========")
    print_matrix(hist_to_count2[0],0,1)
    print("========    Final post-flat BR histogram  ==========")
    print_matrix(hist_to_count2[1],0,1)


    print("======== SIGNAL PROBS ==========")
    print_matrix(probs_to_count2[0],3,1)
    print("========    BR PROBS  ==========")
    print_matrix(probs_to_count2[1],3,1)


    return
       
## Plot samples in pt (or variable of choice) and return a list of probabilities for keeping events
def getProbabilities(h5Dir, sampleTypes, year, setType, bins, binSize, maxRange, flattenIndex, mass_type):
    print("Begin making probabilities array")
    probs = [] # First axis is sampleTypes, second axis is ptBins, values are probability to keep event in sample,ptBin
    binnedNEvents = [] # First axis is sampleTypes, second axis is ptBins, values are number of events in sample,ptBin

    ## The following block should populate the binnedNEvents list
        
    for sample in sampleTypes:
        print("Processing", year, sample, mass_type)
        mass_str= mass_type + "_"
        sample_str = sample + "_"


        filePath = h5Dir+sample_str+'Sample_'+ mass_str +year+'_BESTinputs'
        
        if setType == "":
            filePath = filePath+".h5"
            #filePath = filePath+"_flattened.h5"
        else:
            filePath = filePath+"_"+setType+".h5"
            #filePath = filePath+"_"+setType+"_flattened.h5"

        print("Looking for file: %s"%filePath)
        f = h5py.File(filePath, 'r')
        binnedNEvents.append([])

        ## Only needs to be done on smallest key, BEST_vars
        ## Output shape of myData is (NEvents,)
        myData = np.array(f["BES_vars"][...,flattenIndex])
        print("myData", myData.shape)
        # print("Begin bin looping")
        for currLowRange in bins:
            currHighRange = min(currLowRange+binSize, maxRange)
            ## Pick out data in bin
            myDataBool = (currLowRange<myData)*(myData<currHighRange)
            ## (bool = True -> mask) so need to invert mask to keep desired info
            dataMask = ma.masked_array(myData, mask=~myDataBool)
            ## Invert mask again (Maybe this could be cleaner)
            ## Shape of truncated data is (NEventsPass,)
            myTruncatedData = dataMask[~dataMask.mask]
            ## Append NEvents in bin to last element (list) of binnedNEvents, i.e. sample
            binnedNEvents[-1].append(len(myTruncatedData))
    #print(binnedNEvents)

    ## Convert to numpy array to better manipulate
    ## binnedNEvents is shape (nSamples, nBins, 1) with values NEventsInBinForSample
    binnedNEvents = np.array(binnedNEvents)
    print(binnedNEvents.shape)
    print("First entry", binnedNEvents[0])
    
    ## Next, populate probs which is a list of shape (NSamples, NBins, 1) with value keepProbability
    print("Begin making prob calculations")
    for sampleIndex in range(0, len(sampleTypes)):
        binnedProbs = []
        for binIndex in range(0, len(bins)):
            print(binnedNEvents[...,binIndex])
            num = float(min(binnedNEvents[...,binIndex]))
            denom = float(binnedNEvents[sampleIndex][binIndex])
            if denom > 0:
                binnedProbs.append(num/denom)
            else:
                binnedProbs.append(0.)
        probs.append(binnedProbs)
    print (probs)
    return probs 

## return 2D probability to flatten wrt two vars (here HT and SJ mass)
### WARNING: this assumes only two categories: sig and BR. This is not meant for to be used for arbitrary categories (though could be adapted)

def getProbabilities2D(h5Dir, sampleTypes, year, setType, mass_type,args):

    ## init vars to use 
    HT_max = args.rangeHighHT
    HT_min = args.rangeLowHT

    SJ_mass_max = args.rangeHighSJMass
    SJ_mass_min = args.rangeLowSJMass

    N_bins_HT = args.nBinsHT # represents HT
    N_bins_SJ = args.nBinsSJMass  # represents SJ mass

    nEvents = []
    probs  = []

    HT_index      = 111 # change to the index of HT
    SJ_mass_index = 88 # change to index of SJ mass

    for sample in sampleTypes:
        ### get sig N events
        print("Getting events for %s."%sample)
        filePath = h5Dir+ "%s_"%sample +'Sample_all_mass_'+year+'_BESTinputs'+"_"+setType+".h5"

        print("Looking for file: %s" % filePath)
        f = h5py.File(filePath, 'r')
        if f: print("Found file: %s" % filePath)
        else: 
            print("ERROR finding file: %s" % filePath)
            return

        HT_values    = np.array(f["BES_vars"][...,HT_index])
        HT_values.astype(float)
        SJ_mass_values= np.array(f["BES_vars"][...,SJ_mass_index])
        SJ_mass_values.astype(float)
        # Create 2D histogram for SJ mass and HT
        hist2d, _, _ = np.histogram2d(HT_values, SJ_mass_values, bins=[N_bins_HT,N_bins_SJ], range=[[HT_min, HT_max], [SJ_mass_min, SJ_mass_max]])
        nEvents.append(hist2d.T)
    
    nEvents = np.array(nEvents) # this is a 2xNxM matrix (2x (NxM) for {sig,BR}x{HT}x{SJ mass})
    nEvents = nEvents.astype(float)

    #print("Number of signal events: ")
    #print_matrix(nEvents[0],0,1)

    #print("Number of background events: ")
    print_matrix(nEvents[1],0,1)



    print("Calculating probabilities.")
    min_events = np.minimum(nEvents[0], nEvents[1])  # gives NxM array of the minimum events between samples
    min_events = np.array(min_events).astype(float)
    probs.append(  min_events/nEvents[0] )
    probs.append(  min_events/nEvents[1] )

    probs = np.nan_to_num(probs)

    print("Signal probs is: ")
    print_matrix(probs[0],6,1)
    
    print("BR probs is: ")
    print_matrix(probs[1],6,1)

    with open("test_probs.pkl", "wb") as f:
        pickle.dump(probs, f)

    return probs ##  [signal_probs, BR_probs]

def print_matrix(matrix, precision=2, cell_width=6):
    """
    Pretty-prints a 2D list or NumPy array like a matrix in Python 2.

    Parameters:
    - matrix: 2D list or NumPy array
    - precision: Number of decimal places for floats
    - cell_width: Width of each printed cell
    """
    fmt_float = "%%%d.%df" % (cell_width, precision)
    for row in matrix:
        formatted_row = " ".join(
            fmt_float % elem if isinstance(elem, float)
            else str(elem).rjust(cell_width)
            for elem in row
        )
        print formatted_row

## helper function for testing: write (to a pkl) the number of events in each 2D bin for both samples AFTER flattening
def printPostFlatNEvents(h5Dir, sampleTypes, year, setType, mass_type,args, bins_HT, bins_SJ):
    
    ## init vars to use 
    HT_max = args.rangeHighHT
    HT_min = args.rangeLowHT

    SJ_mass_max = args.rangeHighSJMass
    SJ_mass_min = args.rangeLowSJMass

    N_bins_HT = args.nBinsHT # represents HT
    N_bins_SJ = args.nBinsSJMass  # represents SJ mass

    nEvents = []
    probs  = []

    HT_index      = 111 # change to the index of HT
    SJ_mass_index = 88 # change to index of SJ mass

    for sample in sampleTypes:

        nEvents.append(np.zeros( (N_bins_HT, N_bins_SJ) ))

        ### get sig N events
        print("Getting events for %s."%sample)
        filePath = h5Dir+ "%s_"%sample +'Sample_all_mass_'+year+'_BESTinputs'+"_"+setType+"_flattened.h5"

        print("Looking for file: %s" % filePath)
        fIn = h5py.File(filePath, 'r')
        if fIn: print("Found file: %s" % filePath)
        else: 
            print("ERROR finding file: %s" % filePath)
            return

        #HT_values  = np.array(f["BES_vars"][...,HT_index])
        #HT_values.astype(float)
        #SJ_mass_values= np.array(f["BES_vars"][...,SJ_mass_index])
        #SJ_mass_values.astype(float)


        batch_num = 1
        counter = 0
        totalEvents = fIn[list(fIn.keys())[0]].shape[0]
        userBatchSize = args.batchSize


        # Count total events in batches
        while counter < totalEvents:
            batchSize = userBatchSize if (totalEvents > counter + userBatchSize) else (totalEvents - counter)
            
            # Grab SJ mass and HT data for the current batch
            SJ_mass_values = np.array(fIn["BES_vars"][counter:counter + batchSize, 88])
            HT_values = np.array(fIn["BES_vars"][counter:counter + batchSize, 111])

            hist2d, _, _ = np.histogram2d( HT_values, SJ_mass_values, bins=[N_bins_HT,N_bins_SJ], range=[[HT_min, HT_max], [SJ_mass_min, SJ_mass_max]])
            
            nEvents[sampleTypes.index(sample)] += hist2d.T

            #print("%s ------> Batch number %s"%(sample,batch_num))
            #print("Number of POST-FLAT %s events: "%(sample))
            #print_matrix(nEvents[sampleTypes.index(sample)],0,1)

            batch_num+=1
            counter += batchSize


        """# Create 2D histogram for SJ mass and HT
        hist2d, _, _ = np.histogram2d(HT_values, SJ_mass_values, bins=[N_bins_HT,N_bins_SJ], range=[[HT_min, HT_max], [SJ_mass_min, SJ_mass_max]])
        nEvents.append(hist2d)"""
    
    nEvents = np.array(nEvents) # this is a 2xNxM matrix (2x (NxM) for {sig,BR}x{HT}x{SJ mass})
    nEvents = nEvents.astype(float)

    print("==============================")
    print("============ DONE ============")
    print("==============================")

    print("Number of POST-FLAT signal events: ")
    print_matrix(nEvents[0],0,1)

    print("Number of POST-FLAT background events: ")
    print_matrix(nEvents[1],0,1)

    return

# Main function should take in arguments and call the functions you want
# -s is the samples to process: if 'all' then it does QCD,W,Z,Top,b,Higgs. Else you can provide a comma separated list
# -st is the types of sample sets to process, i.e. train, validation, test. If 'all' then it does these three but also the pre-split samples
# -b is the batch size to do the copying when flattening. This is a performance hyper-parameter. The output is unaffected.
# -fi flattenIndex is the BESvars index to flatten on. Currenltly the default is 548 since that corresponds to pt in the current samples.
# -rl rangeLow is the lower limit to set the bins. Anything below this will always be rejected.
# -rh rangeHigh is the upper limit to set the bins. Anything above this will always be rejected.
# -nb is the number of bins for the flattening range. Bin size is set by (rl-rh)/nbins.
# -hd is the path to the location of the input h5 files
# -o is the path to the location where to send the outputs
# -d enables the debugging flag with all print statements
if __name__ == "__main__":
    
    # Take in arguments
    parser = argparse.ArgumentParser(description='Parse user command-line arguments to execute format conversion to prepare for training.')
    parser.add_argument('-y', '--years',
                        dest='years',
                        help='<Required> Which (comma separated) years to process. Examples: 1) all; 2) 2016,2018',
                        default='all')
    parser.add_argument('-st', '--setTypes',
                        dest='setTypes',
                        help='<Required> Which (comma separated) set types to process. Examples: 1) --all (includes pre-split); 2) train,validation,test',
                        default='all')
    parser.add_argument('-b', '--batchSize',
                        dest='batchSize',
                        type=int,
                        default=250000)

    parser.add_argument( '--nBinsSJMass',
                        dest='nBinsSJMass',
                        type=int,
                        # default=175)
                        # default=55)
                        default=50)
    parser.add_argument('--nBinsHT',
                        dest='nBinsHT',
                        type=int,
                        # default=175)
                        # default=55)
                        default=50)
    parser.add_argument('-hd','--h5Dir',
                        dest='h5Dir',
                        default="h5samples/")
    parser.add_argument('-o','--outDir',
                        dest='outDir',
                        default="h5samples/")
    parser.add_argument('-d','--debug',
                        action='store_true')
    parser.add_argument('--rangeLowHT',
                        dest='rangeLowHT',
                        type=float,
                        # default=0)
                        default=1400)
    parser.add_argument('--rangeHighHT',
                        dest='rangeHighHT',
                        type=float,
                        # default=3500)
                        # default=1600)
                        default=10000)

    parser.add_argument('--rangeLowSJMass',
                        dest='rangeLowSJMass',
                        type=float,
                        # default=0)
                        default=0)
    parser.add_argument('--rangeHighSJMass',
                        dest='rangeHighSJMass',
                        type=float,
                        # default=3500)
                        # default=1600)
                        default=6500)

    args = parser.parse_args()

    if not args.years == "all": years = args.years.split(',')
    if not args.setTypes == "all": setTypes = args.setTypes.split(',')
    if args.debug:
        print("Samples to process: ", sampleTypes)
        print("Sets to process: ", setTypes)
        print("Years to process: ", years)
        #print("Flattenning Index: ", args.flattenIndex)
        print("Reading Every nEvents: ", args.batchSize)

    # Make directories you need
    if not os.path.isdir(args.outDir): os.mkdir(args.outDir)

    binSize_HT = (args.rangeHighHT-args.rangeLowHT)/args.nBinsHT
    bins_HT = [args.rangeLowHT+binSize_HT*i for i in range(0,args.nBinsHT+1)]
    
    binSize_SJ = (args.rangeHighSJMass-args.rangeLowSJMass)/args.nBinsSJMass
    bins_SJ = [args.rangeLowSJMass+binSize_SJ*i for i in range(0,args.nBinsSJMass+1)]

    if args.debug: print("Range: ", args.nBins," bins, from ", bins[0], " to ", bins[len(bins)-1]+binSize, " in steps of ", binSize) 
    if args.debug: print("Rejecting events above: ", args.rangeHigh)
    
    for decays_type in decays_types:
        if decays_type == "allDecays":
            # sampleTypes = ["allDecays","Top","QCD", "WJets"]
            sampleTypes = ["allDecays","bg"]
        else: sampleTypes = sampleTypes_

        for mass_type in mass_types:
            for year in years:
                for setType in setTypes:
                    keepnEvents2D = getProbabilities2D(args.h5Dir, sampleTypes, year, setType, mass_type,args)
                    flattenFile(keepnEvents2D, args.h5Dir, args.outDir, sampleTypes, year, setType, bins_SJ, bins_HT, args.batchSize, mass_type, args)
                    printPostFlatNEvents(args.h5Dir, sampleTypes, year, setType, mass_type,args, bins_HT, bins_SJ)




