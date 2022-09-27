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

setTypes = ["test", "validation", "train"]
sampleTypes = ["TT"]
h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"

eventDict = {
    "test":{"TT":[714, 8737, 81478, 171957, 230835, 234291, 241185] },
    "validation":{"TT":[146336, 303291] },
    "train":{"TT":[8254, 72276, 76237, 79337, 203215, 240008, 265060, 274909, 337512, 339778, 341118, 405994, 444094, 479755, 480741, 495579, 691807, 696461, 702410, 763895, 773389, 919411, 1248911, 1259892, 1335612, 1345437, 1415713, 1503674, 1589421, 1751346, 1902370, 1909324, 1941872, 2064217, 2074979, 2332581, 2333309, 2336994, 2494392, 2570140, 2574698, 2657903, 2658606, 2662556, 2670105, 2740596, 2779438, 2821766, 2895710, 2907899, 2948961, 2969778, 2975145, 2986754] }
}

for setType, partDict in eventDict.items():
    print(setType)
    for part, events in partDict.items():
        print("Trimming: ", part, len(events))
        # events.sort(key=int)
        # badString = []
        # for event in events: badString.append(str(event))
        
        f  = h5py.File(h5Dir+part+"Sample_2017_BESTinputs_"+setType+"_flattened.h5","r+")
        oldShape = f["BES_vars"].shape[0]
        
        # print(oldShape)
        mask = []
        for i in range(oldShape): mask.append(i)
            # if not str(i) in badString: mask.append(i) 
        # print(len(mask))        

        for key in f.keys():
            if key == "BES_vars": continue
            print(key)

            dset = f[key]
            # oldShape = dset.shape
            # mask = [False if str(i) in badString else True for i in range(oldShape[0])]

            # ds = dset[mask]
            ds = dset[()]
            ds = ds[mask]
            dset.resize(ds.shape[0], axis=0)
            dset[:] = ds
            del ds
            

print("\nUpdated all datasets!")

