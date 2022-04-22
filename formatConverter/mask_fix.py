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

TopFrame = ["Top"]
HiggsFrame = ["Higgs"]
WFrame = ["W"]
oldFrames = TopFrame + HiggsFrame + WFrame
ZFrame = ["Z"]
BottomFrame = ["Bottom"]
ak8noSoftFrame = ["ak8"]
ak8SoftFrame = ["ak8SoftDrop"]
ak8Frames = ak8noSoftFrame + ak8SoftFrame
boost50  = ["50GeV"]
boost100 = ["100GeV"]
boost150 = ["150GeV"]
boost200 = ["200GeV"]
boost250 = ["250GeV"]
boost300 = ["300GeV"]
boost350 = ["350GeV"]
boost400 = ["400GeV"]
boosts = boost50 + boost100 + boost150 + boost200 + boost250 + boost300 + boost350 + boost400
# baseFrames = oldFrames + boost3
baseFrames = ak8noSoftFrame +  WFrame + boost300
allFrames = boosts + oldFrames + BottomFrame + ZFrame + ak8Frames
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


# frames = oldFrames + BottomFrame + ak8Frames
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
# varFile = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/old/fix/allBESTVars.txt"
varFile = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/formatConverter/masks/fixBESTMask_allFrames.txt"
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
                if "isotropy" in var: continue
                if "nJets" in var: continue
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
# masksToMake = {}

# masksToMake["Z"] = baseFrames + ZFrame
# masksToMake["B"] = baseFrames + BottomFrame
# masksToMake["ZB"] = baseFrames + ZFrame + BottomFrame
# masksToMake["noSoft"] = baseFrames + ak8noSoftFrame
# masksToMake["Soft"] = baseFrames + ak8SoftFrame
# masksToMake["ak8"] = baseFrames + ak8Frames
# masksToMake["200"] = baseFrames + boost2
# masksToMake["400"] = baseFrames + boost4
# masksToMake["200400"] = baseFrames + boost2 + boost4
# masksToMake[""] = baseFrames +
# masksToMake[""] = baseFrames +

# for boost in boosts:
# for boost in boost3:
    # frames = oldFrames + ak8Frames + [boost] + ZFrame + BottomFrame

    # frames = oldFrames + ak8noSoftFrame + [boost] 
    # frames = oldFrames + ak8noSoftFrame + [boost] + ZFrame
    # frames = oldFrames + ak8noSoftFrame + [boost] + BottomFrame
    # frames = oldFrames + ak8noSoftFrame + [boost] + ZFrame + BottomFrame


masksToMake = {
    # "WZ":ZFrame+WFrame,
    # "TopW":TopFrame+WFrame,
    # "TopHiggs":TopFrame+HiggsFrame,
    # "TopWZ":TopFrame+ZFrame+WFrame,
    # "WHiggs":WFrame+HiggsFrame,
    # "WZHiggs":WFrame+HiggsFrame+ZFrame,
    # "bothAK8":ak8SoftFrame + ak8noSoftFrame,
    # "300Top":boost300+ TopFrame,
    # "300Higgs":boost300+ HiggsFrame,
    # "300W":boost300+ WFrame,
    # "300Z":boost300+ZFrame,
    # "300TopHiggs":boost300+TopFrame+HiggsFrame,
    # "300TopW":boost300+TopFrame+WFrame,
    # "300WHiggs":boost300+WFrame+HiggsFrame,
    # "300ak8":boost300+ak8noSoftFrame,
    # "300ak8SoftDrop":boost300+ak8SoftFrame,
    # "300bothAK8":boost300+ak8noSoftFrame+ak8SoftFrame,
    # "Wak8":WFrame+ak8noSoftFrame,
    # "Wak8SoftDrop":WFrame+ak8SoftFrame,
    # "WbothAK8":WFrame+ak8noSoftFrame+ak8SoftFrame,
    # "Topak8":TopFrame+ak8noSoftFrame,
    # "Topak8SoftDrop":TopFrame+ak8SoftFrame,
    # "TopbothAK8":TopFrame+ak8noSoftFrame+ak8SoftFrame,
    # "Higgsak8":HiggsFrame+ak8noSoftFrame,
    # "300Topak8":boost300+TopFrame+ak8noSoftFrame,
    # "Higgsak8SoftDrop":HiggsFrame+ak8SoftFrame,
    # "HiggsbothAK8":HiggsFrame+ak8noSoftFrame+ak8SoftFrame
    # "HiggsTopW":HiggsFrame+WFrame+TopFrame

    # "300Wak8":boost300+ak8noSoftFrame+WFrame,
    # "300Wak8SD":boost300+ak8SoftFrame+WFrame,
    # "300Wbothak8":boost300+ak8Frames+WFrame,

    # "300Wak8Higgs":boost300+ak8noSoftFrame+WFrame+HiggsFrame,
    # "300Wak8Top":boost300+ak8noSoftFrame+WFrame+TopFrame,
    # "300Wak8HiggsTop":boost300+ak8noSoftFrame+WFrame+HiggsFrame+TopFrame,
    # "300Wak8200":boost300+ak8noSoftFrame+WFrame+boost200,

    # "300Wak8SDHiggs":boost300+ak8SoftFrame+WFrame+HiggsFrame,
    # "300Wak8SDTop":boost300+ak8SoftFrame+WFrame+TopFrame,
    # "300Wak8SDHiggsTop":boost300+ak8SoftFrame+WFrame+HiggsFrame+TopFrame,
    # "300Wak8SD200":boost300+ak8SoftFrame+WFrame+boost200,

    # "300Wbothak8Higgs":boost300+ak8Frames+WFrame+HiggsFrame,
    # "300Wbothak8Top":boost300+ak8Frames+WFrame+TopFrame,
    # "300Wbothak8HiggsTop":boost300+ak8Frames+WFrame+HiggsFrame+TopFrame,
    # "300Wbothak8200":boost300+ak8Frames+WFrame+boost200,

    # "300Wak8Higgs400":boost300+ak8noSoftFrame+WFrame+HiggsFrame+boost400,
    # "300Wak8Top400":boost300+ak8noSoftFrame+WFrame+TopFrame+boost400,
    # "300Wak8HiggsTop400":boost300+ak8noSoftFrame+WFrame+HiggsFrame+TopFrame+boost400,
    # "300Wak8200400":boost300+ak8noSoftFrame+WFrame+boost200+boost400,

    "300WHT400":boost300+WFrame+HiggsFrame+TopFrame+boost400,
    "300Wak8HT400":boost300+ak8noSoftFrame+WFrame+HiggsFrame+TopFrame+boost400,
    "300Wak8SDHT400":boost300+ak8SoftFrame+WFrame+HiggsFrame+TopFrame+boost400,
    "300Wbothak8HT400":boost300+ak8Frames+WFrame+HiggsFrame+TopFrame+boost400,

    "300WHT":boost300+WFrame+HiggsFrame+TopFrame,
    "300Wak8HT":boost300+ak8noSoftFrame+WFrame+HiggsFrame+TopFrame,
    "300Wak8SDHT":boost300+ak8SoftFrame+WFrame+HiggsFrame+TopFrame,
    "300Wbothak8HT":boost300+ak8Frames+WFrame+HiggsFrame+TopFrame,

    "300ZHT400":boost300+ZFrame+HiggsFrame+TopFrame+boost400,
    "300Zak8HT400":boost300+ak8noSoftFrame+ZFrame+HiggsFrame+TopFrame+boost400,
    "300Zak8SDHT400":boost300+ak8SoftFrame+ZFrame+HiggsFrame+TopFrame+boost400,
    "300Zbothak8HT400":boost300+ak8Frames+ZFrame+HiggsFrame+TopFrame+boost400,


    # "allboosts":boosts,
    # "50150250350":boost50+boost150+boost250+boost350,
    # "100200300400":boost100+boost200+boost300+boost400,

    # "allboosts":boosts+ak8noSoftFrame,
    # "50150250350ak8":boost50+boost150+boost250+boost350+ak8noSoftFrame,
    # "100200300400ak8":boost100+boost200+boost300+boost400+ak8noSoftFrame,


    }
for suffix, frames in masksToMake.items():
    writeVars = []
    for var in frameStripVars:
        for frame in frames:

            if var[0] == '_': writeVars.append(frame + "Frame" + var)
            else:             writeVars.append(var + "_" + frame)
    writeVars = writeVars + labVars
    writeVars.sort()

    newMask = "masks/indv/" + suffix + ".txt"
    # newMask = "masks/fixBESTMask_" + suffix + ".txt"
    # newMask = "masks/newBESTMask_noZ.txt"
    with open(newMask, mode='wt') as f:
        for var in writeVars:
            if ("nJets" in var) or ("isotropy" in var):
                if not "Higgs" in var: continue
            if "deep" in var: continue
            f.write(varDict[var] + ":" + var + '\n')


# for frame in allFrames:
#     writeVars = []
#     for var in frameStripVars:
#         if var[0] == '_': writeVars.append(frame + "Frame" + var)
#         else:             writeVars.append(var + "_" + frame)
#     writeVars = writeVars + labVars
#     writeVars.sort()

#     newMask = "masks/indv/" + frame + ".txt"
#     # newMask = "masks/newBESTMask_noZ.txt"
#     with open(newMask, mode='wt') as f:
#         for var in writeVars:
#             # if ("nJets" in var) or ("isotropy" in var):
#             #     if not "Higgs" in var: continue
#             if "deep" in var: continue
#             f.write(varDict[var] + ":" + var + '\n')