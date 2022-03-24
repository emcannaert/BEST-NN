#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# mask.py /////////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Sam Abbott -----------------------------------------------------------
# This script generates masks to be used in the training step. ////////////////////
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Figure out how this script should fit into the final release (will we keep it?).
# Does this belong in formatConverter or training?
# If we keep it, it needs to be improved and needs more comments.

import os

oldFrames = ["Top", "Higgs", "W"]
zFrame = ["Z"]
bottomFrame = ["Bottom"]
ak8noSoftFrame = ["ak8"]
ak8SoftFrame = ["ak8SoftDrop"]
ak8Frames = ak8noSoftFrame + ak8SoftFrame
boosts = ["50GeV", "100GeV", "150GeV", "200GeV", "250GeV", "300GeV", "350GeV", "400GeV"]
boost2 = ["200GeV"]
boost3 = ["300GeV"]
boost4 = ["400GeV"]
baseFrames = oldFrames + boost3
allFrames = boosts + oldFrames + bottomFrame + zFrame + ak8Frames
"""
maskDir = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/old/50/fix/"
fixDir = maskDir + "noak8/"
if not os.path.isdir(fixDir): os.makedirs(fixDir)
files = os.listdir(maskDir)

for file in files:
    if not file == "newBESTMask_300.txt": continue
    # if file == "fix": continue
    # if file == "no_trim": continue
    # if file == "noak8": continue
    # # if file == "50": continue
    # if "Z" in file: continue
    # if "_B" in file: continue
    # if "300" in file: continue

    print(file)
    fileDict = {}
    allVars = []
    with open(maskDir+file) as f:
        for line in f:
            ind, var = line.split(':') # Reads in colon seperated values
            var = var.strip() # Remove '\n'
            if "ak8" in var: continue
            fileDict[ind] = var
            allVars.append(int(ind))
    allVars.sort(key=int)
    with open(fixDir+file, mode='wt') as f:
        for ind in allVars:
            f.write(str(ind) + ":" + fileDict[str(ind)] + '\n')


            # newind = ind
            # if ind >= 350: newind -= 9
            # if ind >= 356: newind -= 5 
            # if ind >= 545: newind -= 16
            # if ind >= 559: newind -= 10
            # if ind >= 565: newind -= 5
            # f.write(str(newind) + ":" + fileDict[str(ind)] + '\n')
    # print("done " + file)
"""
"""
# oldMask = "/uscms/home/bonillaj/nobackup/h5samples_PCTv2/oldBESTMask.txt"
# allOldVars   = []
# frameOldVars = []
# frameStripOldVars = []
# with open(oldMask) as f: # Opens sample file, code will implicitly close file when done with loop
#     for line in f:
#         oldIndex, oldVar = line.split(':') # Reads in colon seperated values
#         oldVar = oldVar.strip() # Remove '\n'
#         allOldVars.append(oldVar)
        
#         for frame in oldFrames:
#             if frame in oldVar: 
#                 frameOldVars.append(oldVar)
#                 stripOldVar = oldVar[:oldVar.rfind("_")] #Strip everything after the final '_'
#                 if stripOldVar not in frameStripOldVars: frameStripOldVars.append(stripOldVar)
# labOldVars = list(set(allOldVars) - set(frameOldVars))


# frames = oldFrames + bottomFrame + ak8Frames
# frames = boosts + ak8Frames
# frames = allFrames
# frames = oldFrames 
# writeVars = []
# for var in frameStripOldVars:
#     for frame in frames:
#         if frame == "Z": continue
#         writeVars.append(var + "_" + frame)
# writeVars = writeVars + labOldVars
# writeVars.sort()
"""

# varFile = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/h5samples/BESvarList.txt"
# Fixed for Basic scale h5 file:
varFile = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/old/fix/allBESTVars.txt"
varDict = {}
allVars = []
frameVars = []
frameStripVars = []
with open(varFile) as f:
    for line in f:
        index, var = line.split(':') # Reads in colon seperated values 
        var = var.strip() # Remove '\n'
        allVars.append(var)
        varDict[var] = index
        for frame in allFrames:
            if frame in var: 
                if "jetAK8" in var: continue # Skip lab vars
                frameVars.append(var)
                if "Frame" in var: stripVar = var[var.find("_"):] #Strip everything before the first '_' (i.e. 'TopFrame_jet_pz1' -> '_jet_pz1')
                else:              stripVar = var[:var.rfind("_")] #Strip everything after the final '_' (i.e. 'sphericity_Top' -> 'sphericity')
                
                if stripVar not in frameStripVars: frameStripVars.append(stripVar)
labVars = list(set(allVars) - set(frameVars))

# frames = oldFrames + ["Bottom", "ak8", "ak8SoftDrop"]
# frames = boosts + ["ak8", "ak8SoftDrop"]
# frames = boosts + oldFrames + ["Bottom", "ak8", "ak8SoftDrop"]
# frames = boosts + oldFrames 
# frames = oldFrames + ak8Frames 
masksToMake = {}
# masksToMake["Z"] = baseFrames + zFrame
# masksToMake["B"] = baseFrames + bottomFrame
# masksToMake["ZB"] = baseFrames + zFrame + bottomFrame
# masksToMake["noSoft"] = baseFrames + ak8noSoftFrame
# masksToMake["Soft"] = baseFrames + ak8SoftFrame
# masksToMake["ak8"] = baseFrames + ak8Frames
masksToMake["200"] = baseFrames + boost2
masksToMake["400"] = baseFrames + boost4
masksToMake["200400"] = baseFrames + boost2 + boost4
# masksToMake[""] = baseFrames +
# masksToMake[""] = baseFrames +

# for boost in boosts:
# for boost in boost3:
    # frames = oldFrames + ak8Frames + [boost] + zFrame + bottomFrame

    # frames = oldFrames + ak8noSoftFrame + [boost] 
    # frames = oldFrames + ak8noSoftFrame + [boost] + zFrame
    # frames = oldFrames + ak8noSoftFrame + [boost] + bottomFrame
    # frames = oldFrames + ak8noSoftFrame + [boost] + zFrame + bottomFrame
for suffix, frames in masksToMake.items():
    writeVars = []
    for var in frameStripVars:
        for frame in frames:

            if var[0] == '_': writeVars.append(frame + "Frame" + var)
            else:             writeVars.append(var + "_" + frame)
    writeVars = writeVars + labVars
    writeVars.sort()

    newMask = "masks/fixBESTMask_" + suffix + ".txt"
    # newMask = "masks/newBESTMask_noZ.txt"
    with open(newMask, mode='wt') as f:
        for var in writeVars:
            if ("nJets" in var) or ("isotropy" in var):
                if not "Higgs" in var: continue
            if "deep" in var: continue
            f.write(varDict[var] + ":" + var + '\n')