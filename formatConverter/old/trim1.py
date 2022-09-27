#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# trimh5.py ///////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Sam Abbott -----------------------------------------------------------
# This script trims some bad events from the BB and QCD h5 files.  ////////////////
# Bad variable is "284:ak8SoftDropFrame_jet_energy0" //////////////////////////////
# The script confirms that the events were removed correctly. /////////////////////
# SoftDropFrame still being tested, could be left out of final release. /////////// 
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Figure out if we should keep this script in the final release.
# If we keep it, it needs to be more general and needs more comments.

import numpy as np
import h5py

# eventDict = { "BB":[2534325, 2935341, 2969700], "QCD":[331219, 833858, 1170782, 1301201, 1306620, 1490148, 1565553] }
eventDict = {"QCD":[331219, 833858, 1170782, 1301201, 1306620, 1490148, 1565553] }

for sample, badEvents in eventDict.items():
    print("\nBeginning " + sample)
    f = h5py.File("/uscms/home/bonillaj/nobackup/h5samples_ULv1/"+sample+"Sample_2017_BESTinputs_train_flattened.h5","r+")
    # f = h5py.File("/uscms/home/msabbott/nobackup/abbott/CMSSW_10_6_27/src/BEST/formatConverter/h5samples/"+sample+"Sample_2017_BESTinputs_train_flattened.h5","r+")
    badEvents.sort(key=int)
    badString = []
    for event in badEvents: badString.append(str(event))
    print(badString)
    oldShape = f["PF_cands_AllFrame"].shape[0]
    # print(oldShape)
    mask = []
    for i in range(oldShape):
        if not str(i) in badString: mask.append(i) 
    print(len(mask))
    # mask = [False if str(i) in badString else True for i in range(oldShape)]
    # print(np.sum(mask))
    # print(mask)
    for key in f.keys():
        if key == "BES_vars": continue
        print(key)
        dset = f[key]
        # oldShape = dset.shape
        # ds = f[key][mask]
        # ds = dset[mask]
        ds = dset[()]
        ds = ds[mask]
        dset.resize(ds.shape[0], axis=0)
        dset[:] = ds
        del ds
        # print("Array shape before trimming: " + str(oldShape) )
        

    # print("There are " + str(len(badEvents)) + " Bad Events:")
    # for event in badEvents:
    #     print("Event, Value:", event, oldDS[event, 284])
    
    # print("Creating trimmed array...")
        # newDS = np.delete(oldDS, badEvents, 0)
        # del oldDS
        # newShape = newDS.shape
    # print("Array shape after trimming: " + str(newShape) )

    # print("Verifying array was trimmed correctly...")
    # if not ( oldShape[0] - newShape[0] == len(badEvents) ):
    #     print("ERROR, events don't match")
    #     quit()
    # elif not ( oldShape[1] == newShape[1] ) :
    #     print("ERROR, variables don't match")
    #     quit()
    # elif np.isnan(newDS[...,284]).any() or np.isposinf(newDS[...,284]).any():
    #     print("ERROR, NaN or +Inf values remain in new dataset")
    #     print("Nan:")
    #     print(np.where(np.isnan(newDS[...,284])))
    #     print("+Inf:")
    #     print(np.where(np.isposinf(newDS[...,284])))
    #     quit()
    # else:
    #     print("Array trimmed correctly! \nUpdating dataset in h5 file...")

    # print("Dataset shape before update: " + str(dset.shape))
    #     dset.resize(newShape[0], axis=0)
    # # print("Dataset shape after resize: " + str(dset.shape))
    #     dset[:] = newDS
    #     del newDS
    # # print("Dataset shape after replacement: " + str(dset.shape))

    # print("Verifying that dataset was updated correctly...")
    # if not ( dset.shape == newShape ) :
    #     print("ERROR, shape is incorrect")
    #     quit()
    # elif np.isnan(dset[...,284]).any() or np.isposinf(dset[...,284]).any():
    #     print("ERROR, NaN or +Inf values remain in new dataset")
    #     print("Nan:")
    #     print(np.where(np.isnan(dset[...,284])))
    #     print("+Inf:")
    #     print(np.where(np.isposinf(dset[...,284])))
    #     quit()
    # else:
    #     print("Dataset updated correctly!!!")
