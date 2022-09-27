import numpy as np
import h5py


sampleTypes = ["WW","ZZ","HH","TT","BB","QCD"]
setTypes = ["validation","test","train"]
# setTypes = ["train"]
h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/"
# h5Keys = ["PF_cands_LabFrame", "PF_cands_AllFrame"]
h5Keys = ['PF_cands_HiggsFrame', 'PF_cands_LabFrame', 'PF_cands_TopFrame', 'PF_cands_WFrame', 'PF_cands_ZFrame', "PF_cands_AllFrame"]
besChunks = h5py.File(h5Dir+"QCDSample_2017_BESTinputs_test_flattened.h5","r")[h5Keys[0]].chunks
besShape =  h5py.File(h5Dir+"QCDSample_2017_BESTinputs_test_flattened.h5","r")[h5Keys[0]].shape
# besChunks = [10, 551]
# put BES variables in data frames
# update this to make the pfCands datasets directly readable, so putting allframes in each one

for mySet in setTypes:
    print(mySet)
    for sample in sampleTypes:
        print(sample)
        # outF = h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + mySet + "_flattened_PF_Cands_LabAllFrame.h5", "w")
        outF = h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + mySet + "_flattened_"+h5Key+".h5", "w")
        for h5Key in h5Keys:
            print(h5Key)
            array = h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + mySet + "_flattened.h5","r")[h5Key]
            batchSize = array.shape[0] // 4
            for i in range(4):
                if i == 0:
                    # print(i)
                    outF.create_dataset(h5Key, data=array[:batchSize,:,:], maxshape=(None, besShape[1], besShape[2]), chunks=(1, besChunks[1], besChunks[2]), compression='lzf', shuffle=True)
                else:
                    begin = batchSize * i
                    if i == 3: end = None
                    else:      end = batchSize * (i+1)
                    outF[h5Key].resize(outF[h5Key].shape[0] + len(array[begin:end]), axis=0)
                    outF[h5Key][-len(array[begin:end]):] = array[begin:end]
                    # print(i, batchSize, begin, end)
        outF.close()

        # outF = h5py.File(h5Dir+sample+"Sample_2017_BESTinputs_" + mySet + "_flattened_"+h5Key+".h5", "r")
        # print(mySet,sample,outF[h5Key].chunks)
print("Finished making datasets")
