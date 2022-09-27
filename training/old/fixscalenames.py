#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# recordScalerParameters.py ///////////////////////////////////////////////////////
#----------------------------------------------------------------------------------
# Author(s): Sam Abbott ///////////////////////////////////////////////////
# This program records the Scaler Parameters for an already existing model /////////
#----------------------------------------------------------------------------------

from sklearn.externals.joblib import load
import os

years = ["2016_APV", "2016", "2017", "2018"]

maskPath = "/uscms/home/msabbott/submitBEST/formatConverter/h5samples/BESvarList.txt"
varDict = {}
with open(maskPath, "r") as f:
    for line in f: 
        index, var = line.split(':')
        var = var.strip()
        varDict[index] = var
        # maskIndex.append(line.split(':')[0])


# print("Mask size: " + str(len(maskIndex)))
print("Mask size: " + str(len(varDict.keys())))

scalerDir = "ScalerParameters_flatTop/"
# scalerDir = "ScalerParameters_flattened/"

for year in years:
    scalePath =  os.path.join(scalerDir, "backup/BESTScalerParameters_"+ year + '.txt')
    newPath = os.path.join(scalerDir, "BESTScalerParameters_"+ year + '.txt')
    with open(newPath, 'w') as fout:
        with open(scalePath, 'r') as fin:
            for line in fin:
                index, scale, param1, param2 = line.split(',')
                param2 = param2.strip()
                var = varDict[index]
                fout.write('{},{},{},{}\n'.format(var, scale, param1, param2))

