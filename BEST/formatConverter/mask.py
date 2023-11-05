#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# mask.py /////////////////////////////////////////////////////////////////////////
#==================================================================================
# Author(s): Mark Samuel Abbott ---------------------------------------------------
# This script generates masks to be used in the training step. ////////////////////
#----------------------------------------------------------------------------------

################################## NOTES TO SELF ##################################
# Figure out how this script should fit into the final release (will we keep it?).
# Does this belong in formatConverter or training?
# If we keep it, it needs to be improved and needs more comments.

import os

oldFrames = ["Top", "Higgs", "Z", "W"]
bottomFrame = ["Bottom"]
ak8Frames = ["ak8", "ak8SoftDrop"]
boosts = ["50GeV", "100GeV", "150GeV", "200GeV", "250GeV", "300GeV", "350GeV", "400GeV"]
allFrames = boosts + oldFrames + bottomFrame + ak8Frames

oldMask = "/uscms/home/bonillaj/nobackup/h5samples_PCTv2/oldBESTMask.txt"
allOldVars   = []
frameOldVars = []
frameStripOldVars = []
with open(oldMask) as f: # Opens sample file, code will implicitly close file when done with loop
    for line in f:
        oldIndex, oldVar = line.split(':') # Reads in colon seperated values
        oldVar = oldVar.strip() # Remove '\n'
        allOldVars.append(oldVar)
        
        for frame in oldFrames:
            if frame in oldVar: 
                frameOldVars.append(oldVar)
                stripOldVar = oldVar[:oldVar.rfind("_")] #Strip everything after the final '_'
                if stripOldVar not in frameStripOldVars: frameStripOldVars.append(stripOldVar)
labOldVars = list(set(allOldVars) - set(frameOldVars))


# frames = oldFrames + bottomFrame + ak8Frames
# frames = boosts + ak8Frames
# frames = allFrames
frames = oldFrames 
writeVars = []
for var in frameStripOldVars:
    for frame in frames:
        if frame == "Z": continue
        writeVars.append(var + "_" + frame)
writeVars = writeVars + labOldVars
writeVars.sort()


varFile = "/uscms_data/d3/cannaert/BEST/CMSSW_10_6_27/src/BEST/formatConverter/h5samples/BESvarList.txt"
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
#         for frame in allFrames:
#             if frame in var: 
#                 if "jetAK8" in var: continue # Skip lab vars
#                 frameVars.append(var)
#                 if "Frame" in var: stripVar = var[var.find("_"):] #Strip everything before the first '_' (i.e. 'TopFrame_jet_pz1' -> '_jet_pz1')
#                 else:              stripVar = var[:var.rfind("_")] #Strip everything after the final '_' (i.e. 'sphericity_Top' -> 'sphericity')
                
#                 if stripVar not in frameStripVars: frameStripVars.append(stripVar)
# labVars = list(set(allVars) - set(frameVars))

# # frames = oldFrames + ["Bottom", "ak8", "ak8SoftDrop"]
# # frames = boosts + ["ak8", "ak8SoftDrop"]
# # frames = boosts + oldFrames + ["Bottom", "ak8", "ak8SoftDrop"]
# # frames = boosts + oldFrames 
# frames = oldFrames
# writeVars = []
# for var in frameStripVars:
#     for frame in frames:
#         if frame == "Z": continue
#         if var[0] == '_': writeVars.append(frame + "Frame" + var)
#         else:             writeVars.append(var + "_" + frame)
# writeVars = writeVars + labVars
# writeVars.sort()

newMask = "masks/oldBESTMask_noZ.txt"
# newMask = "masks/newBESTMask_noZ.txt"
with open(newMask, mode='wt') as f:
    for var in writeVars:
        f.write(varDict[var] + ":" + var + '\n')
