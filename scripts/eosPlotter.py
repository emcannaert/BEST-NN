#=========================================================================================
# eosPlotter.py --------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

# This script was written for Python 2.7.5.
# This file lives in the BEST/scripts directory, but can be ran anywhere.
# This file reads in root files from a eos space and plots them. 
# The eos space, files read, and variables plotted, can all be modified.
# This file creates a plots/ directory.

import sys
import os
import time
import numpy
import subprocess
from datasetDictionary import datasetDict # Import the dictionary of sample files
import ROOT

######################################### NOTES TO SELF ###########################
# Update titles for nostack plots part->var+part
# Using slim jets to cut in run_part.py, collection seems to be different than what we are using (slimmed)????? (why data above 3500Gev pT???????)

#==================================================================================
# Setup ///////////////////////////////////////////////////////////////////////////
#==================================================================================

startTime = time.time() # Tracks how long script takes
ROOT.gROOT.SetBatch(1) # Prevent any windows from being displayed 
ROOT.gErrorIgnoreLevel = ROOT.kWarning # Suppress output lower than ignore level "Warning" (suppresses the Info... plot has been created) 
# ROOT.gStyle.SetPalette(ROOT.kRainBow) #55 kRainBow palette; bad for 2D data but fine here 
# ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette 
ROOT.gStyle.SetTitleAlign(13)
ROOT.gStyle.SetTitleX(0.)
# ROOT.gStyle.SetTitleXSize(0.01)
# ROOT.gStyle.SetTitleFontSize(0.025)

plotPath = "plots/"

# Create the vector of boosts used in BESTProducer:
# boosts.append("ak8SoftDrop")
# boosts.append("ak8")
# iterMass = 50
# while iterMass <= 400: 
#     boosts.append(iterMass) # Add this mass to the vector, then increment by 1 GeV if any condition is true, or 5 GeV if none are true. 
#     iterMass += 50
#     # if ( (110 <= iterMass < 160) or (180 <= iterMass < 220) ):  iterMass += 1
#     # elif                (iterMass < 400):                       iterMass += 5
#     # else:                                                       iterMass += 100

# Create particle dictionary: { particle:[rest mass, mass range, empty root file dictionary], ...}
partDict = { "HH":[{},{}], "WW":[{},{}], "ZZ":[{},{}], "tt":[{},{}], "bb":[{},{}], "QCD":[{},{}] }

# Create variable dictionaries: { variable:binInfo }
# varDict = { "FoxWolfH1_":"100,0,1", "FoxWolfH2_":"100,0,1", "FoxWolfH3_":"100,0,1", "FoxWolfH4_":"100,0,1", "isotropy_":"100,0,1",
#             "sphericity_":"100,0,1", "aplanarity_":"100,0,0.2", "thrust_":"100,0.5,1", "nJets_":"50,0,50", "jet12_mass_":"100,0,300",
#             "jet23_mass_":"100,0,150", "jet13_mass_":"100,0,200", "jet1234_mass_":"100,0,400", "jet12_CosTheta_":"100,-1,1",
#             "jet23_CosTheta_":"100,-1,1", "jet13_CosTheta_":"100,-1,1", "jet1234_CosTheta_":"100,-1,1", "jet12_DeltaCosTheta_":"100,-1,1",
#             "jet13_DeltaCosTheta_":"100,-1,1", "jet23_DeltaCosTheta_":"100,-1,1", "asymmetry_":"100,-1,1"
#           }

labFrameDict ={"jetAK8_pt":"100,500,4500"}
# labFrameDict ={ "jetAK8_mass":"100,0,600", "jetAK8_SoftDropMass":"100,0,250", "jetAK8_charge":"100,-0.5,0.5", "jetAK8_pt":"100,0,4500", "nJets":"8,0,8",
#                 "nSecondaryVertices":"15,0,15", "SV_pt":"100,0,500", "SV_eta":"100,-2.5,2.5", "SV_phi":"100,-3.2,3.2", "SV_mass":"100,0,20",
#                 "SV_nTracks":"20,0,20", "SV_chi2":"100,0,20", "SV_Ndof":"25,0,25", "bDiscSubJet_Max":"100,0,1", "bDiscSubJet_Max_index":"6,0,6",
#                 "bDisc":"100,0,1", "bDisc_probb":"100,0,1", "bDisc_probbb":"100,0,1","bDisc1":"100,0,1", "bDisc1_probb":"100,0,1", 
#                 "bDisc1_probbb":"100,0,1", "bDisc2":"100,0,1", "bDisc2_probb":"100,0,1", "bDisc2_probbb":"100,0,1"
#                 # "bDisc":"100,-2,1", "bDisc_probb":"100,-1,1", "bDisc_probbb":"100,-1,1","bDisc1":"100,-2,1", "bDisc1_probb":"100,-1,1", 
#                 # "bDisc1_probbb":"100,-1,1", "bDisc2":"100,-2,1", "bDisc2_probb":"100,-1,1", "bDisc2_probbb":"100,-1,1"
#               }
                                
print("Checking directory for root files...")

# Gather and organize mass points
for part, subDicts in partDict.items():
    massPoints = datasetDict["mc"]["2017"][part].keys()
    if part == "tt" or part == "QCD":   massPoints.sort() # Mass points not easily sorted for QCD (#to#) and TT (M_#_W_#)
    else:                               massPoints.sort(key=int) # key=int necessary to sort 500 before 5000, etc.
    subDicts.append(massPoints)

#==================================================================================
# Create Chains and Stacks ////////////////////////////////////////////////////////
#==================================================================================

# Use eosls to get root files, then create chains and stacks for them
# eospath = "/store/user/msabbott/"
eosls = ["xrdfs", "root://cmseos.fnal.gov", "ls", "-u"] # Alias for eosls, broken up into pieces for subprocess 
for part, subDicts in partDict.items():
    chainDict =  subDicts[0]
    stackDict =  subDicts[1]
    massPoints = subDicts[2]
    for massPoint in massPoints: # Iterate over each unique dataset by mass point
        crabInfo = datasetDict["mc"]["2017"][part][massPoint] 
        endIndex = crabInfo[1].find('Run') # Identify eos dir name
        eospath = ["/store/user/msabbott" + crabInfo[1][:endIndex] + "crab_" + crabInfo[0]] # Example: /store/user/msabbott/BulkGravToWWToWhadWhad_narrow_M-3000_TuneCP5_13TeV-madgraph-pythia/crab_GravitonWW_3000GeV_trees
        timeStamp = subprocess.check_output(eosls + eospath).split("\n") # This should give the timestamps, code works for only one timestamp (can edit)
        if len(timeStamp) > 2: # Check that there is only one timestamp. Final entry of list should always be empty string from the .split("\n"): [timestamp, ""]
            print("Error, more than one timestamp detected for:\n" + str(timeStamp) ) 
            sys.exit(1)
        startIndex = timeStamp[0].rfind("/") # Identify timestamp
        eospath[0] += timeStamp[0][startIndex:] + "/0000/" # Add the timestamp to eospath, should lead to BESTInputs*.root files now
        files = subprocess.check_output(eosls + eospath ).split("\n") # Will list dir containing BESTInputs, among other things.

        # Now identify the BESTInputs and add to TChain dictionary
        chainDict[massPoint] = ROOT.TChain("run/jetTree")
        chain = chainDict[massPoint] 
        for file in files:
            if "BEST" in file: chain.Add(file)

    for labvar in labFrameDict.keys(): # This iterates over the lab frame variables in the root file
        stackDict[labvar] = ROOT.THStack(part,part) # THStack object for this lab variable (fill with all mass points for this particle)



#==================================================================================
# Plot Lab Frame Variables ////////////////////////////////////////////////////////
#==================================================================================

print("Plotting lab frame variables...")

for labvar, binInfo in labFrameDict.items(): # This iterates over the lab frame variables in the root file
    finalStack = ROOT.THStack(labvar,labvar)
    
    allPath = plotPath+"all/"+labvar+"/"
    if not os.path.exists(allPath): os.makedirs(allPath) # If directory doesn't exist, create it        

    for part, subDicts in partDict.items():
        chainDict =  subDicts[0]
        stack =      subDicts[1][labvar]
        massPoints = subDicts[2]

        indvPath = plotPath+"indvidual/"+labvar+"/"+part+"/"
        if not os.path.exists(indvPath): os.makedirs(indvPath) # If directory doesn't exist, create it        

        # hsum = ROOT.TH1F(labvar+"_"+part,labvar+"_"+part,binInfo)
        hsum = ROOT.TH1F(labvar+"_"+part,labvar+"_"+part,100,500.,4500.) # Cant get THStack to display just final sum of histos. Manually summing histos, to add the sums to a stack

        ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette for indvidual plots
        for massPoint in massPoints:
            chain = chainDict[massPoint]

            # chain.Draw(labvar+">>htemp("+binInfo+")", "", "norm")
            chain.Draw(labvar+">>htemp("+binInfo+")", "")
            htemp = ROOT.gROOT.FindObject("htemp") # Grab histogram, it is named "htemp" by default, and overwritten each time

            hsum.Add(htemp)

            # Save individual plots
            title = part+"_"+labvar+"_"+massPoint
            canvas = ROOT.TCanvas(title)
            canvas.cd()
            htemp.SetTitle(title)
            # htemp.SetStats(0) # Hide stats box
            htemp.Draw("HIST") # HIST needed bc of normalization
            canvas.SaveAs(indvPath+title+".png") # Save plot as png 
            canvas.Close() # Close canvas now that we are done

            htemp.SetTitle(massPoint) # Change hist title to the current mass point, which updates the legend entry later
            stack.Add(htemp.Clone(massPoint)) # Clone histogram, add to Stack for current labvar

            del htemp # Delete htemp to keep memory usage low
        
        # Plot all particles together:
        canvas = ROOT.TCanvas(labvar) # Create canvas
        canvas.cd() # Switch to new canvas

        # Intermission: draw summed histo and add to final stack
        hsum.Draw("HIST")
        canvas.SaveAs(allPath+labvar+"_"+part+"_sum.png") # Save plot as png 
        hsum.SetTitle(part)
        finalStack.Add(hsum.Clone(part))

        # For each particle, plot all mass points separately on the same plot
        ROOT.gStyle.SetPalette(ROOT.kRainBow) # kRainBow palette for all together plots; bad for 2D data but fine here 
        # Now redraw, then plot all mass points separetely for this particle
        stack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
        legend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords, top right of canvas
        legend.SetNColumns(7) # Set legend columns
        ROOT.gPad.Update() # Draw legend
        canvas.SaveAs(allPath+labvar+"_"+part+".png") # Save plot as png 
        canvas.Close() # Close canvas now that we are done
    
    # Plot all summed histos together 
    canvas = ROOT.TCanvas(labvar) # Create canvas
    canvas.cd() # Switch to new canvas    
    finalStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate, HIST needed bc of normalization
    legend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords, top right
    legend.SetNColumns(3) # Set legend columns
    ROOT.gPad.Update() # Draw legend
    canvas.SaveAs(allPath+labvar+"_all.png") # Save plot as png 
    canvas.Close() # Close canvas now that we are done

# Check how long the script took to run
runf = open("timeLog", "w") 
timeTaken = divmod(time.time() - startTime, 60.)
runf.write("Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.")
runf.close