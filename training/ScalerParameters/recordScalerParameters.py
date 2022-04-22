#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# recordScalerParameters.py ///////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Sam Abbott ///////////////////////////////////////////////////
# This program records the Scaler Parameters for an already existing model /////////
#----------------------------------------------------------------------------------

from sklearn.externals.joblib import load

maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/fixBESTMask_ak8.txt"
maskIndex = []
with open(maskPath, "r") as f:
    for line in f: 
        maskIndex.append(line.split(':')[0])
print("Mask size: " + str(len(maskIndex)))

scalerPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/ScalerParameters/newBEST_Basic.joblib"
ct = load(scalerPath)

scalePath = 'ScalerParameters/' + "BESTParameters" + '.txt'
print("Saving Parameters: " + scalePath)
with open(scalePath, 'w') as f:
    for name, transformer, events in ct.transformers_: 
        numEvents = len(events)
        print(name)
        if "noscale" in name:
            nameKey = "NoScale"
            param1 = [0]*numEvents
            param2 = [0]*numEvents
        else:    
            if "min" in name:
                nameKey = "MinMax"
                # param2 = transformer.min_
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
            if str(event) not in maskIndex: continue
            f.write('{},{},{},{}\n'.format(event, nameKey, param1[i], param2[i]))
    