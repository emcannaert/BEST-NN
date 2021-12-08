import ROOT
import sys
import os
import time
import numpy

startTime = time.time()

# cmslpc179
ROOT.gROOT.SetBatch(1) # Prevent any windows from being displayed
ROOT.gErrorIgnoreLevel = ROOT.kWarning # Suppress output lower than ignore level "Warning" (suppresses the Info... plot has been created)
# ROOT.gStyle.SetPalette(ROOT.kRainBow) #55 kRainBow palette; bad for 2D data but fine here 
# ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette 
ROOT.gStyle.SetTitleAlign(13)
ROOT.gStyle.SetTitleX(0.)

plotPath = "plots/"
scanDir = "noCut/"
"""
The job splitting of this task is 'Automatic', please refer to this FAQ for a description of the jobs status summary:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/CRAB3FAQ#What_is_the_Automatic_splitting
More details of Automatic Splitting process for this task (including possible failures) are in the Dagman Log files in:
https://cmsweb.cern.ch:8443/scheddmon/0197/cms110/211008_150541:maabbott_crab_QCD_Pt_1400to1800_trees/AutomaticSplitting/

Jobs status:                    failed                   90.1% (1125/1249)
                                finished                  9.9% ( 124/1249)

No publication information (publication has been disabled in the CRAB configuration file)

Error Summary: (use crab status --verboseErrors for details about the errors)

1038 jobs failed in postprocessing step

  17 jobs failed with exit code 50664

   6 jobs failed with exit code 60307

   5 jobs failed with exit code 8033

   4 jobs failed with exit code 8021

   3 jobs failed with exit code 50115

   1 jobs failed with exit code 8028

Could not find exit code details for 51 jobs.

Have a look at https://twiki.cern.ch/twiki/bin/viewauth/CMSPublic/JobExitCodes for a description of the exit codes.
"""
# Create the vector of boosts used in BESTProducer:
# boosts = [15,75,80,81,90,180]
# boosts = [1]
boosts = []
boosts.append("ak8")
boosts.append("ak8_SoftDrop")
iterMass = 5
while iterMass <= 1000: 
    boosts.append(iterMass) # Add this mass to the vector, then increment by 1 GeV if any condition is true, or 5 GeV if none are true. 
    # iterMass += 5 # Skip the fine points
    if ( (110 <= iterMass < 160) or (180 <= iterMass < 220) ):  iterMass += 1
    elif                (iterMass < 400):                       iterMass += 5
    else:                                                       iterMass += 100

# Create dictionary. Add variables and particles to the initial dictionary/list as desired
# initialPartDict = { "bb":[10., 4.]}
# initialPartDict = { "HH":[125., 12.], "WW":[80.,6]}
initialPartDict = {"HH":[120.,5.,{}], "WW":[80.,10.,{}], "ZZ":[90.,10.,{}], "tt":[170.,10.,{}], "bb":[5.,30.,{}], "QCD":[100.,10.,{}]}
print("Checking directory for root files...")
files = os.listdir(os.getcwd())
for part, partValues in initialPartDict.items():    
    for file in files:
        if (part in file) and ("BESTInputs.root" in file): 
            # Store root file, with mass point as the key
            endIndex = file.find("_BEST")
            partValues[2][file[:endIndex]] = [file] 

# initialVarDict = {variable:{} for variable in ["FoxWolfH1_", "FoxWolfH2_"] }
initialVarDict = {variable:{} for variable in [
                                            # "SV_pt_", "SV_eta_", "SV_phi_", "SV_mass_", "SV_nTracks_", "SV_chi2_", "SV_Ndof_", "nSecondaryVertices_"
                                            "FoxWolfH1_", "FoxWolfH2_", "FoxWolfH3_", "FoxWolfH4_", "isotropy_", "sphericity_", "aplanarity_", 
                                            "thrust_", "nJets_", "jet12_mass_", "jet23_mass_", "jet13_mass_", "jet1234_mass_", "jet12_CosTheta_",
                                            "jet23_CosTheta_", "jet13_CosTheta_", "jet1234_CosTheta_", "jet12_DeltaCosTheta_", "jet13_DeltaCosTheta_", 
                                            "jet23_DeltaCosTheta_", "asymmetry_"
                                             ] }
Dictionary = { "PARTICLES":initialPartDict , "VARIABLES":initialVarDict } # Will append root things to the particle dict, and will replace the values in the variable dict several times

# labFrameVars = ["jetAK8_mass", "jetAK8_SoftDropMass", "SV_pt", "SV_eta", "SV_phi", "SV_mass", "SV_nTracks", "SV_chi2", "SV_Ndof", "nSecondaryVertices"]
labFrameDict = {labvar:{} for labvar in [
                                    "jetAK8_mass", "jetAK8_SoftDropMass", "jetAK8_charge", "jetAK8_pt", "nJets", 
                                    "nSecondaryVertices", "SV_pt", "SV_eta", "SV_phi", "SV_mass", "SV_nTracks", "SV_chi2", "SV_Ndof", 
                                    "bDisc", "bDisc_probb", "bDisc_probbb","bDisc1", "bDisc1_probb", "bDisc1_probbb",
                                    "bDisc2", "bDisc2_probb", "bDisc2_probbb", "bDiscSubJet_Max", "bDiscSubJet_Max_index"
                                        ] }
ak8MassDict = {}
# Iterate over dictionary values containing rest mass [0] and mass scan range [1] for each particle, and add root things:
print("Opening root files...")
for part, partValues in Dictionary["PARTICLES"].items():
    # partValues.append(ROOT.TFile.Open(partValues[2])) # TFile [3] for each particle
    for partmass, partmassValues in partValues[2].items():
        partmassValues.append(ROOT.TFile.Open(partmassValues[0])) # Open root file [1] for each particle for each mass point
        partmassValues.append( {} ) # Empty dictionary [2] to fill
        for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
            partmassValues[2][var+partmass+"_Coarse_Scan"] = ROOT.THStack(var+partmass+"_Coarse_Scan",var+partmass+"_Coarse_Scan") # Coarse Scan Stack [3][key] for each particle for each variable
            partmassValues[2][var+partmass+"_Fine_Scan"]   = ROOT.THStack(var+partmass+"_Fine_Scan",var+partmass+"_Fine_Scan")    # Fine Scan Stack [3][key] for each particle for each variable
        ak8MassDict[partmass] = ROOT.THStack(partmass,partmass)

# print(Dictionary)
# Now particle dictionary has the form: 
# {particle:[restmass, restmass range, TFile object, {variable_particle_Coarse_Scan:THStack object, variable_particle_Fine_Scan:TH1F object}] }
# While the variable dictionary has the form: {variable: {} }

# Plot lab frame variables, only one each per particle
print("Plotting lab frame variables...")
ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette for indvidual plots
for labvar in labFrameDict: # This iterates over the lab frame variables in the root file
    if (labvar == "jetAK8_mass") or (labvar == "jetAK8_SoftDropMass"): labPath = plotPath+"lab/ak8_masses/"
    else:                                                              labPath = plotPath+"lab/"+labvar+"/"
    if not os.path.exists(labPath): os.makedirs(labPath) # If directory doesn't exist, create it
    labStack = ROOT.THStack(labvar,labvar) # THStack object for this lab variable (fill with 6 particles)
    
    for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
        for partmass, partmassValues in partValues[2].items():
            jettree = partmassValues[1].Get("run/jetTree") # Get jet tree from the TFile for this particle. Root file structure is part+"_BESTInputs.root/run/jetTree/[leaves]"

            if labvar == "nSecondaryVertices":  jettree.Draw(labvar+">>htemp(15,0,15)", "", "norm") # Create histogram
            elif labvar == "nJets":             jettree.Draw(labvar+">>htemp(8,0,8)", "", "norm") # Create histogram
            else:                               jettree.Draw(labvar, "", "norm") # Create histogram
            htemp = ROOT.gROOT.FindObject("htemp") # Grab histogram, it is named "htemp" by default, and overwritten each time
            if labvar == "nSecondaryVertices": htemp.SetNdivisions(15)
            
            htemp.SetTitle(partmass) # Change hist title to the current particle, which updates the all legend entry later
            labStack.Add(htemp.Clone(partmass)) # Clone histogram, add to Stack for current labvar
            
            # Save individual plots
            # if (labvar == "jetAK8_mass") or (labvar == "jetAK8_SoftDropMass"): 
            #     htemp.SetTitle(labvar)
            #     ak8MassDict[partmass].Add(htemp.Clone(partmass)) # Add to ak8 mass plots (plot masses together)
            # else:
            #     indvLabString = labvar+"_"+partmass
            #     indvLabCanvas = ROOT.TCanvas(indvLabString)
            #     indvLabCanvas.cd()
            #     htemp.SetTitle(indvLabString)
            #     # htemp.SetStats(0) # Hide stats box
            #     htemp.Draw("HIST") # HIST needed bc of normalization
            #     indvLabCanvas.SaveAs(labPath+indvLabString+".png") # Save plot as png 
            #     indvLabCanvas.Close() # Close canvas now that we are done

            del htemp # Delete htemp to keep memory usage low
            
        # End particle loop
        ROOT.gStyle.SetPalette(ROOT.kRainBow) # kRainBow palette for all together plots; bad for 2D data but fine here 

        # Plot all particles together:
        labCanvas = ROOT.TCanvas(labvar) # Create canvas
        labCanvas.cd() # Switch to new canvas
        labStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
        if labvar == "nSecondaryVertices": labStack.GetHistogram().SetNdivisions(15)
        if (labvar == "jetAK8_mass") or (labvar == "jetAK8_SoftDropMass"): labStack.GetXaxis().SetRangeUser(0,600)
        labLegend = ROOT.gPad.BuildLegend(0.5,1.,1.) # Create legend at coords
        labLegend.SetNColumns(5) # Set legend columns to 3
        ROOT.gPad.Update() # Draw legend
        if (labvar == "jetAK8_mass") or (labvar == "jetAK8_SoftDropMass"): saveString = plotPath+"lab/ak8_masses/"+labvar+"_all.png"
        else:                                                              saveString = plotPath+"lab/"+labvar+"/"+labvar+"_all.png"
        labCanvas.SaveAs(saveString) # Save plot as png 
        labCanvas.Close() # Close canvas now that we are done

# ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette for indvidual plots
# for partmass, stack in ak8MassDict.items():
#     ak8Canvas = ROOT.TCanvas("ak8_Masses_"+partmass) # Create canvas
#     ak8Canvas.cd() # Switch to new canvas
#     stack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
#     stack.GetXaxis().SetRangeUser(0,600)
#     ak8legend = ROOT.gPad.BuildLegend(0.5,1.,1.) # Create legend at coords
#     ak8legend.SetNColumns(2) # Set legend columns to 2
#     ROOT.gPad.Update() # Draw legend
#     ak8Canvas.SaveAs(plotPath+"lab/ak8_masses/"+partmass+".png") # Save plot as png 
#     ak8Canvas.Close() # Close canvas now that we are done
    


# Two sets of plots: 
# Boost set, which plots all particles for each boost (everything needs to be reset each loop)
# Coarse/Fine set, which plots many boosts for each particle (setup before loop, fill through whole loop, draw after loop)

# get hist, add fit before clone. plots might be a mess after...
# some sort of get bin content, iterate from left to right, if it goes below threshold save the x value or the bin, then set the range?
# could use getMax or getMin to iterate over less bins?
# add vars

print("Plotting boosted variables...")
ROOT.gStyle.SetPalette(ROOT.kRainBow) # kRainBow palette for all together plots; bad for 2D data but fine here 

for boost in boosts: # Iterate over boosts
    print("Boost: "+str(boost)+" GeV")

    for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        if (boost == "ak8") or (boost == "ak8_SoftDrop"): boostString = var+boost
        else:                                             boostString = var+str(boost)+"GeV" # Name of the histogram in each root file for this variable for this boost
        boostStack = ROOT.THStack(boostString,boostString) # THStack object for this variable for this boost (fill with 6 particles for each mass point)

        for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
            # Assign dictionary entries to variables for readability:
            restMass    = partValues[0] # Rest mass of each particle
            massRange   = partValues[1] # Range from rest mass to examine for fine scan

            for partmass, partmassValues in partValues[2].items():
                jettree     = partmassValues[1].Get("run/jetTree") # Get jet tree from the TFile for this particle. Root file structure is part+"_BESTInputs.root/run/jetTree/[leaves]"
                coarseStack = partmassValues[2][var+partmass+"_Coarse_Scan"] # THStack object for Coarse Scan for this variable for this particle
                fineStack   = partmassValues[2][var+partmass+"_Fine_Scan"] # THStack object for Fine Scan for this variable for this particle
                
                # # Conditions that trigger the fine and coarse scans:
                # if (boost == "ak8") or (boost == "ak8_SoftDrop"):
                #     trigCoarse = True
                #     trigFine   = True
                # else:
                #     trigCoarse = True if             (boost > 10) and ((boost % 15) == 0)            else False # Fill Coarse Scan every 15 GeV, starting with 30 GeV (so 30, 45, 60,...)
                #     trigFine   = True if  (restMass - massRange) <= boost <= (restMass + massRange)  else False # If boost is near rest mass, fill Fine Scan

                # Skip loop if not doing boosts and coarse or fine scan wont trigger:
                # if not ( trigCoarse or trigFine ): continue

                #1-condition to inverse
                # varValues[2] = jettree.Draw(varValues[0], "(jetAK8_mass>85)*(jetAK8_mass<95)") # Leaves [2] where variable data are stored by name [0]
                if var == "nJets_": jettree.Draw(boostString+">>htemp(50,0,50)", "", "norm") # Create histogram
                else:               jettree.Draw(boostString, "", "norm") # Create histogram
                htemp = ROOT.gROOT.FindObject("htemp") # Grab histogram, it is named "htemp" by default, and overwritten each time
                if var == "aplanarity_": htemp.GetXaxis().SetRangeUser(0,0.2)
                # if var == "nSecondaryVertices_": htemp.SetNdivisions(15)
                
                htemp.SetTitle(partmass) # Change hist title to the current particle, which updates Boost legend entry later
                boostStack.Add(htemp.Clone(partmass)) # Clone histogram, add to Stack for current boost
                # Conditions that trigger the fine and coarse scans:
                if (boost == "ak8") or (boost == "ak8_SoftDrop"):
                    htemp.SetTitle(boost)
                    fineStack.Add(htemp.Clone(str(boost)))
                    coarseStack.Add(htemp.Clone(str(boost)))
                else:
                    htemp.SetTitle(str(boost)+"GeV") # Now change the title to current boost, which updates Scan legend entries later
                    if             (boost > 10) and ((boost % 15) == 0):            coarseStack.Add(htemp.Clone(str(boost)+"GeV")) # Fill Coarse Scan every 15 GeV, starting with 30 GeV (so 30, 45, 60,...)
                    if  (restMass - massRange) <= boost <= (restMass + massRange):  fineStack.Add(htemp.Clone(str(boost)+"GeV")) # If boost is near rest mass, fill Fine Scan
                
                # Save individual plots
                # indvPath = plotPath+"boosts/"+var[:-1]+"/"+partmass+"/"
                # if not os.path.exists(indvPath): os.makedirs(indvPath) # If directory doesn't exist, create it

                # indvString = boostString+"_"+partmass
                # indvCanvas = ROOT.TCanvas(indvString)
                # indvCanvas.cd()
                # htemp.SetTitle(indvString)
                # # htemp.SetStats(0) # Hide stats box
                # htemp.Draw("HIST") # HIST needed bc of normalization
                # indvCanvas.SaveAs(indvPath+indvString+".png") # Save plot as png 
                # indvCanvas.Close() # Close canvas now that we are done

                del htemp # Delete htemp to keep memory usage low

            # End particle mass point loop
        # End particle loop

        # Plot boost stuff:
        boostPath = plotPath+"boosts/"+var[:-1]+"/all/" # Unique directory for each variable, trimming the underscore
        if not os.path.exists(boostPath): os.makedirs(boostPath) # If directory doesn't exist, create it

        boostCanvas = ROOT.TCanvas(boostString) # Create canvas
        boostCanvas.cd() # Switch to new canvas
        boostStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
        if var == "aplanarity_": boostStack.GetHistogram().SetAxisRange(0.,0.2)
        # if var == "nSecondaryVertices_": boostStack.GetHistogram().SetNdivisions(15)
        blegend = ROOT.gPad.BuildLegend(0.5,1.,1.) # Create legend at coords
        blegend.SetNColumns(5) # Set legend columns to 2
        ROOT.gPad.Update() # Draw legend
        boostCanvas.SaveAs(boostPath+boostString+"_all.png") # Save plot as png 
        boostCanvas.Close() # Close canvas now that we are done

    # End variable loop
# End boost loop

# Save Scan plots (several boosts on one plot for each particle for each variable):
for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
    for partmass, partmassValues in partValues[2].items():
        for var, varValues in Dictionary["VARIABLES"].items(): # Iterate over each variable

            scanPath = plotPath+"scans/"+scanDir+var[:-1]+"/" # Unique directory for each variable, trimming the underscore
            if not os.path.exists(scanPath): os.makedirs(scanPath) # If directory doesn't exist, create it

            coarseStack = partmassValues[2][var+partmass+"_Coarse_Scan"]
            coarseCanvas = ROOT.TCanvas(var+partmass+"_Coarse_Scan") # Create canvas
            coarseCanvas.cd() # Switch to new canvas
            coarseStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
            if var == "aplanarity_": coarseStack.GetHistogram().SetAxisRange(0.,0.2)
            # if var == "aplanarity_": coarseStack.GetXaxis().SetRangeUser(0,0.2)
            coarselegend = ROOT.gPad.BuildLegend(0.5,1.,1.) # Create legend at coords
            coarselegend.SetNColumns(5) # Set legend columns to 3
            ROOT.gPad.Update() # Draw legend
            coarseCanvas.SaveAs(scanPath+var+partmass+"_Coarse_Scan"+".png") # Save plot as png
            coarseCanvas.Close() # Close canvas now that we are done

            fineStack = partmassValues[2][var+partmass+"_Fine_Scan"] 
            fineCanvas = ROOT.TCanvas(var+partmass+"_Fine_Scan") # Create canvas
            fineCanvas.cd() # Switch to new canvas
            fineStack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
            if var == "aplanarity_": fineStack.GetHistogram().SetAxisRange(0.,0.2)
            # if var == "aplanarity_": fineStack.GetXaxis().SetRangeUser(0,0.2)
            finelegend = ROOT.gPad.BuildLegend(0.5,1.,1.) # Create legend at coords
            finelegend.SetNColumns(5) # Set legend columns to 3
            ROOT.gPad.Update() # Draw legend
            fineCanvas.SaveAs(scanPath+var+partmass+"_Fine_Scan"+".png") # Save plot as png
            fineCanvas.Close() # Close canvas now that we are done
        partmassValues[3].Close() # Close all of the root files

print("Pngs complete!")
print("Creating gifs...")
# want gif of all 6 for each boost traveling from 1 to 200 for each var -> use boost pics already made 
# also that but for individual for each particle for each var -> need to save more boost pics
# Could simplify loop by adding gifs to dict to do less loops. could be integrated into main loop.
Dictionary["PARTICLES"]["all"] = [None,None,{"all":None}]
for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
    print("Variable: " + var[:-1])
    gifPath = plotPath+"boosts/"+var[:-1]+"/" # Unique directory for each variable, trimming the underscore

    for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
        for partmass, partmassValues in partValues[2].items():
            gifFile = gifPath+var+partmass+".gif"
            if os.path.exists(gifFile): os.remove(gifFile) # Delete old gif file
            if not os.path.exists(gifPath+partmass): continue
            for boost in boosts: # Iterate over boosts
                if (boost == "ak8") or (boost == "ak8_SoftDrop"): gifString = var+boost
                elif               ( (boost % 5) != 0 ):          continue
                else:                                             gifString = var+str(boost)+"GeV" # Name of the histogram in each root file for this variable for this boost    
                partImage = ROOT.TImage.Open(gifPath+partmass+"/"+gifString+"_"+partmass+".png")
                if boost != 1000:    partImage.WriteImage(gifFile+"+25") # +50 is 50 centiseconds delay
                else:                partImage.WriteImage(gifFile+"++150++") # The final image needs to be written with the ++ to get it to infinite loop
                del partImage

# print("Gifs complete!")
# Check how long the script took to run
runf = open("timeLog", "w") 
timeTaken = divmod(time.time() - startTime, 60.)
runf.write("Script took "+ str( int(timeTaken[0]) ) + "m " + str( int(timeTaken[1]) ) + "s to complete.")
runf.close