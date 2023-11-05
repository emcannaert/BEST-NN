#=========================================================================================
# localPlotter.py ------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Mark Samuel Abbott ----------------------------------------------------------
#-----------------------------------------------------------------------------------------

import ROOT
import sys
import os
import time
import numpy

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
ROOT.gStyle.SetTitleFontSize(0.025)

plotPath = "fixedbDisc_plots/"
scanDir = "noCut/"

# Create the vector of boosts used in BESTProducer:
boosts = []
boosts.append("ak8SoftDrop")
boosts.append("ak8")
iterMass = 5
while iterMass <= 1000: 
    boosts.append(iterMass) # Add this mass to the vector, then increment by 1 GeV if any condition is true, or 5 GeV if none are true. 
    # iterMass += 5 # Skip the fine points
    if ( (110 <= iterMass < 160) or (180 <= iterMass < 220) ):  iterMass += 1
    elif                (iterMass < 400):                       iterMass += 5
    else:                                                       iterMass += 100

# Create particle dictionary: { particle:[rest mass, mass range, empty root file dictionary], ...}
particleDict = {"HH":[120.,5.,{}], "WW":[80.,10.,{}], "ZZ":[90.,10.,{}], "tt":[170.,10.,{}], "bb":[5.,30.,{}], "QCD":[100.,10.,{}]}

# Create variable dictionaries: { variable:binInfo, .... }
varDict = { "FoxWolfH1_":"100,0,1", "FoxWolfH2_":"100,0,1", "FoxWolfH3_":"100,0,1", "FoxWolfH4_":"100,0,1", "isotropy_":"100,0,1",
            "sphericity_":"100,0,1", "aplanarity_":"100,0,0.2", "thrust_":"100,0.5,1", "nJets_":"50,0,50", "jet12_mass_":"100,0,300",
            "jet23_mass_":"100,0,150", "jet13_mass_":"100,0,200", "jet1234_mass_":"100,0,400", "jet12_CosTheta_":"100,-1,1",
            "jet23_CosTheta_":"100,-1,1", "jet13_CosTheta_":"100,-1,1", "jet1234_CosTheta_":"100,-1,1", "jet12_DeltaCosTheta_":"100,-1,1",
            "jet13_DeltaCosTheta_":"100,-1,1", "jet23_DeltaCosTheta_":"100,-1,1", "asymmetry_":"100,-1,1"
          }

labFrameDict ={ "jetAK8_mass":"100,0,600", "jetAK8_SoftDropMass":"100,0,250", "jetAK8_charge":"100,-0.5,0.5", "jetAK8_pt":"100,0,4500", "nJets":"8,0,8",
                "nSecondaryVertices":"15,0,15", "SV_pt":"100,0,500", "SV_eta":"100,-2.5,2.5", "SV_phi":"100,-3.2,3.2", "SV_mass":"100,0,20",
                "SV_nTracks":"20,0,20", "SV_chi2":"100,0,20", "SV_Ndof":"25,0,25", "bDiscSubJet_Max":"100,0,1", "bDiscSubJet_Max_index":"6,0,6",
                "bDisc":"100,0,1", "bDisc_probb":"100,0,1", "bDisc_probbb":"100,0,1","bDisc1":"100,0,1", "bDisc1_probb":"100,0,1", 
                "bDisc1_probbb":"100,0,1", "bDisc2":"100,0,1", "bDisc2_probb":"100,0,1", "bDisc2_probbb":"100,0,1"
                # "bDisc":"100,-2,1", "bDisc_probb":"100,-1,1", "bDisc_probbb":"100,-1,1","bDisc1":"100,-2,1", "bDisc1_probb":"100,-1,1", 
                # "bDisc1_probbb":"100,-1,1", "bDisc2":"100,-2,1", "bDisc2_probb":"100,-1,1", "bDisc2_probbb":"100,-1,1"
              }
                               
                                    
print("Checking directory for root files...")

files = os.listdir(os.getcwd())
for particle, particleValues in particleDict.items():    
    for file in files:
        if (particle in file) and ("BESTInputs.root" in file): 
            # Store root file, with mass point as the key
            endIndex = file.find("_BEST")
            particleValues[2][file[:endIndex]] = [file] 

print("Opening root files...")
# Iterate over dictionary values containing rest mass [0] and mass scan range [1] for each particle, and add root things:
ak8HistDict = {}
for particle, particleValues in particleDict.items():
    for massPoint, massPointValues in particleValues[2].items():
        massPointValues.append(ROOT.TFile.Open(massPointValues[0])) # Open root file [1] for each particle for each mass point
        massPointValues.append( {} ) # Empty dictionary [2] to fill
        for var, varValues in varDict.items(): # This iterates over the variables in the root file
            massPointValues[2][var+massPoint+"_Coarse_Scan"] = ROOT.THStack(var+massPoint+"_Coarse_Scan",var+massPoint+"_Coarse_Scan") # Coarse Scan Stack [3][key] for each particle for each variable
            massPointValues[2][var+massPoint+"_Fine_Scan"]   = ROOT.THStack(var+massPoint+"_Fine_Scan",var+massPoint+"_Fine_Scan")    # Fine Scan Stack [3][key] for each particle for each variable
        ak8HistDict[massPoint] = ROOT.THStack(massPoint,massPoint)

# Now particle dictionary has the form: {particle:[restmass, restmass range, {mass point:[root file name string, root TFile object, 
#                                                                           {coarse key:THStack Coarse scan object, fine key:THStack Fine scan object} ] } ], ... }
# While the boost variable dictionary has the form: {variable: {}, ... }

#==================================================================================
# Plot Lab Frame Variables ////////////////////////////////////////////////////////
#==================================================================================

# Plot lab frame variables, only one each per particle
print("Plotting lab frame variables...")
ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette for indvidual plots
for labvar, binInfo in labFrameDict.items(): # This iterates over the lab frame variables in the root file
    if (labvar == "jetAK8_mass") or (labvar == "jetAK8_SoftDropMass"): labPath = plotPath+"lab/ak8_masses/"
    else:                                                              labPath = plotPath+"lab/"+labvar+"/"
    if not os.path.exists(labPath): os.makedirs(labPath) # If directory doesn't exist, create it
    labStack = ROOT.THStack(labvar,labvar) # THStack object for this lab variable (fill with 6 particles)
    
    for particle, particleValues in particleDict.items(): # Iterate over each particle
        for massPoint, massPointValues in particleValues[2].items():
            jettree = massPointValues[1].Get("run/jetTree") # Get jet tree from the TFile for this particle. Root file structure is particle+"_BESTInputs.root/run/jetTree/[leaves]"

            jettree.Draw(labvar+">>htemp("+binInfo+")", "", "norm")

            htemp = ROOT.gROOT.FindObject("htemp") # Grab histogram, it is named "htemp" by default, and overwritten each time
            
            htemp.SetTitle(massPoint) # Change hist title to the current particle, which updates the all legend entry later
            labStack.Add(htemp.Clone(massPoint)) # Clone histogram, add to Stack for current labvar
            
            # Save individual plots
            if (labvar == "jetAK8_mass") or (labvar == "jetAK8_SoftDropMass"):
                htemp.SetTitle(labvar)
                ak8HistDict[massPoint].Add(htemp.Clone(labvar))
            else:
                indvLabString = labvar+"_"+massPoint
                indvLabCanvas = ROOT.TCanvas(indvLabString)
                indvLabCanvas.cd()
                htemp.SetTitle(indvLabString)
                # htemp.SetStats(0) # Hide stats box
                htemp.Draw("HIST") # HIST needed bc of normalization
                indvLabCanvas.SaveAs(labPath+indvLabString+".png") # Save plot as png 
                indvLabCanvas.Close() # Close canvas now that we are done

            del htemp # Delete htemp to keep memory usage low
        # End mass point loop
    # End particle loop


    ROOT.gStyle.SetPalette(ROOT.kRainBow) # kRainBow palette for all together plots; bad for 2D data but fine here 
    # Plot all particles together:
    labCanvas = ROOT.TCanvas(labvar) # Create canvas
    labCanvas.cd() # Switch to new canvas
    labStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
    if labvar == "nSecondaryVertices": labStack.GetHistogram().SetNdivisions(15)
    if (labvar == "jetAK8_mass") or (labvar == "jetAK8_SoftDropMass"): labStack.GetXaxis().SetRangeUser(0,600)
    labLegend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords
    labLegend.SetNColumns(5) # Set legend columns
    ROOT.gPad.Update() # Draw legend
    saveString = labPath+labvar+"_all.png"
    labCanvas.SaveAs(saveString) # Save plot as png 
    labCanvas.Close() # Close canvas now that we are done

ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette for indvidual plots
for massPoint, stack in ak8HistDict.items():
    ak8Canvas = ROOT.TCanvas("ak8_Masses_"+massPoint) # Create canvas
    ak8Canvas.cd() # Switch to new canvas
    stack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
    stack.GetXaxis().SetRangeUser(0,600)
    ak8legend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords
    ak8legend.SetNColumns(2) # Set legend columns 
    ROOT.gPad.Update() # Draw legend
    ak8Canvas.SaveAs(plotPath+"lab/ak8_masses/"+massPoint+".png") # Save plot as png 
    ak8Canvas.Close() # Close canvas now that we are done
    
del ak8HistDict
del labFrameDict

#==================================================================================
# Plot Boost Frame Variables //////////////////////////////////////////////////////
#==================================================================================

# Two sets of plots: 
# Boost set, which plots all particles for each boost (everything needs to be reset each loop)
# Coarse/Fine set, which plots many boosts for each particle (setup before loop, fill through whole loop, draw after loop)
# Individual plots are also saved

print("Plotting boosted variables...")
ROOT.gStyle.SetPalette(ROOT.kRainBow) # kRainBow palette for all together plots; bad for 2D data but fine here 

for boost in boosts: # Iterate over boosts
    print("Boost: "+str(boost)+" GeV")

    for var, binInfo in varDict.items(): # This iterates over the variables in the root file
        if type(boost) == type( int() ) :   boostString = var+str(boost)+"GeV"
        else:                               boostString = var+boost
        boostStack = ROOT.THStack(boostString,boostString) # THStack object for this variable for this boost (fill with 6 particles for each mass point)

        for particle, particleValues in particleDict.items(): # Iterate over each particle
            # Assign dictionary entries to variables for readability:
            restMass    = particleValues[0] # Rest mass of each particle
            massRange   = particleValues[1] # Range from rest mass to examine for fine scan

            for massPoint, massPointValues in particleValues[2].items():
                jettree     = massPointValues[1].Get("run/jetTree") # Get jet tree from the TFile for this particle. Root file structure is particle+"_BESTInputs.root/run/jetTree/[leaves]"
                coarseStack = massPointValues[2][var+massPoint+"_Coarse_Scan"] # THStack object for Coarse Scan for this variable for this particle
                fineStack   = massPointValues[2][var+massPoint+"_Fine_Scan"] # THStack object for Fine Scan for this variable for this particle
                
                jettree.Draw(boostString+">>htemp("+binInfo+")", "", "norm")
                htemp = ROOT.gROOT.FindObject("htemp") # Grab histogram, it is named "htemp" by default, and overwritten each time
                
                htemp.SetTitle(massPoint) # Change hist title to the current particle, which updates Boost legend entry later
                boostStack.Add(htemp.Clone(massPoint)) # Clone histogram, add to Stack for current boost

                # Conditions that trigger the fine and coarse scans:
                if type(boost) == type( int() ) :
                    htemp.SetTitle(str(boost)+"GeV") # Now change the title to current boost, which updates Scan legend entries later
                    if             (boost > 10) and ((boost % 15) == 0):            coarseStack.Add(htemp.Clone(str(boost)+"GeV")) # Fill Coarse Scan every 15 GeV, starting with 30 GeV (so 30, 45, 60,...)
                    if  (restMass - massRange) <= boost <= (restMass + massRange):  fineStack.Add(htemp.Clone(str(boost)+"GeV")) # If boost is near rest mass, fill Fine Scan
                else:
                    htemp.SetTitle(boost)
                    fineStack.Add(htemp.Clone(boost))
                    coarseStack.Add(htemp.Clone(boost))
                
                # Save individual plots
                indvPath = plotPath+"boosts/"+var[:-1]+"/"+massPoint+"/"
                if not os.path.exists(indvPath): os.makedirs(indvPath) # If directory doesn't exist, create it

                indvString = boostString+"_"+massPoint
                indvCanvas = ROOT.TCanvas(indvString)
                indvCanvas.cd()
                htemp.SetTitle(indvString)
                # htemp.SetStats(0) # Hide stats box
                htemp.Draw("HIST") # HIST needed bc of normalization
                indvCanvas.SaveAs(indvPath+indvString+".png") # Save plot as png 
                indvCanvas.Close() # Close canvas now that we are done

                del htemp # Delete htemp to keep memory usage low
            # End particle mass point loop
        # End particle loop

        # Plot boost stuff:
        boostPath = plotPath+"boosts/"+var[:-1]+"/all/" # Unique directory for each variable, trimming the underscore
        if not os.path.exists(boostPath): os.makedirs(boostPath) # If directory doesn't exist, create it

        boostCanvas = ROOT.TCanvas(boostString) # Create canvas
        boostCanvas.cd() # Switch to new canvas
        boostStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
        blegend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords
        blegend.SetNColumns(5) # Set legend columns
        ROOT.gPad.Update() # Draw legend
        boostCanvas.SaveAs(boostPath+boostString+"_all.png") # Save plot as png 
        boostCanvas.Close() # Close canvas now that we are done
        del boostStack
    # End variable loop
# End boost loop

print("Boosts complete. Plotting scans...")

#==================================================================================
# Plot Boost Scans ////////////////////////////////////////////////////////////////
#==================================================================================

# These will replace the very large dictionaries, which will be deleted
massPointList = ["all"]
varList = varDict.keys()

# Save Scan plots (several boosts on one plot for each particle for each variable):
for particle, particleValues in particleDict.items(): # Iterate over each particle
    for massPoint, massPointValues in particleValues[2].items():
        massPointList.append(massPoint)

        for var in varDict.keys(): # Iterate over each variable

            scanPath = plotPath+"scans/"+scanDir+var[:-1]+"/" # Unique directory for each variable, trimming the underscore
            if not os.path.exists(scanPath): os.makedirs(scanPath) # If directory doesn't exist, create it

            coarseStack = massPointValues[2][var+massPoint+"_Coarse_Scan"]
            coarseCanvas = ROOT.TCanvas(var+massPoint+"_Coarse_Scan") # Create canvas
            coarseCanvas.cd() # Switch to new canvas
            coarseStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
            coarselegend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords
            coarselegend.SetNColumns(5) # Set legend columns
            ROOT.gPad.Update() # Draw legend
            coarseCanvas.SaveAs(scanPath+var+massPoint+"_Coarse_Scan"+".png") # Save plot as png
            coarseCanvas.Close() # Close canvas now that we are done
            del coarseStack

            fineStack = massPointValues[2][var+massPoint+"_Fine_Scan"] 
            fineCanvas = ROOT.TCanvas(var+massPoint+"_Fine_Scan") # Create canvas
            fineCanvas.cd() # Switch to new canvas
            fineStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
            finelegend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords
            finelegend.SetNColumns(5) # Set legend columns
            ROOT.gPad.Update() # Draw legend
            fineCanvas.SaveAs(scanPath+var+massPoint+"_Fine_Scan"+".png") # Save plot as png
            fineCanvas.Close() # Close canvas now that we are done
            del fineStack
        massPointValues[1].Close() # Close all of the root files
        print(massPointValues[0]+" closed.")    

print("All pngs complete!")
del particleDict
del varDict

#==================================================================================
# Generate Gifs ///////////////////////////////////////////////////////////////////
#==================================================================================

print("Creating gifs...")

for var in varList: # This iterates over the variables in the root file
    print("Variable: " + var[:-1])
    gifPath = plotPath+"boosts/"+var[:-1]+"/" # Unique directory for each variable, trimming the underscore

    for massPoint in massPointList:
        gifFile = gifPath+var+massPoint+".gif"
        if os.path.exists(gifFile): os.remove(gifFile) # Delete old gif file
        if not os.path.exists(gifPath+massPoint): continue
        for boost in boosts: # Iterate over boosts
            if type(boost) == type(int()):
                if ( (boost % 5) != 0 ): continue
                gifString = var+str(boost)+"GeV"
            else: gifString = var+boost

            particleImage = ROOT.TImage.Open(gifPath+massPoint+"/"+gifString+"_"+massPoint+".png")
            if boost != 1000:    particleImage.WriteImage(gifFile+"+25") # +25 is 25 centiseconds delay
            else:                particleImage.WriteImage(gifFile+"++150++") # The final image needs to be written with the ++ to get it to infinite loop
            del particleImage

print("Gifs complete!")

# Check how long the script took to run
runf = open("timeLog", "w") 
timeTaken = divmod(time.time() - startTime, 60.)
runf.write("Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.")
runf.close