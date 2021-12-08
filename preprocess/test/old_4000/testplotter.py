import ROOT
import sys
import os
import time

startTime = time.time()

# cmslpc154

ROOT.gROOT.SetBatch(1) # This prevents anything from being displayed



# dict = { "PARTICLES":{ "HH": [125., 10., tfile, cstring ,ccanvas ,fstring ,fcanvas ], OTHERPARTICLES...}, "VARIABLES": {"FoxWolfH1_":[bstring,canvas,leaf], "FW2":... , "OTHER VARS":... } }
# Create dictionary. Add variables and particles to the initial dictionary/list as desired
# initialPartDict = { "bb":[10., 4.]}
# initialPartDict = { "HH":[125., 12.], "WW":[80.,6]}
initialPartDict = {"HH":[115.,20.], "WW":[65.,15.], "ZZ":[80.,7.], "tt":[155.,11.], "bb":[70.,10.], "QCD":[60.,20]}
# initialVarDict= {variable:[None,None,None] for variable in ["FoxWolfH1_", "FoxWolfH2_", "FoxWolfH3_", "FoxWolfH4_"] }
initialVarDict= {variable:None for variable in ["jetAK8_mass", "jetAK8_SoftDropMass"] }


                                            
# jetAK8_mass

# initialVarDict= {variable:[None,None,None] for variable in ["FoxWolfH1_"] }
Dictionary = { "PARTICLES":initialPartDict , "VARIABLES":initialVarDict } # Will append root things to the particle dict, and will replace the values in the variable dict several times
# Dictionary = { "PARTICLES":initialPartDict , "VARIABLES":      dict.fromkeys( allVariables, [None,None,None]) } # Will append root things to the particle dict, and will replace the values in the variable dict several times

# Iterate over dictionary values containing rest mass [0] and mass scan range [1] for each particle, and add root things:
for part, partValues in Dictionary["PARTICLES"].items():
    partValues.append(ROOT.TFile.Open(part+"_BESTInputs.root")) # TFile [2] for each particle
    partValues.append(ROOT.TCanvas(part)) # Coarse Scan String [3][key] for each particle 

for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
    varValues = ROOT.TCanvas(var) # Create stack [1] for each variable for this boost by name [0]

for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle  
    print(part)
    jettree = partValues[2].Get("run/jetTree") # Get jet tree from the TFile for this particle. Root file structure is part+"_BESTInputs.root/run/jetTree/[leaves]"

    for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        #1-condition to inverse
        
        varValues.append = jettree.Draw(varValues[0]) # Leaves [2] where variable data are stored by name [0]
        varValues.append( ROOT.TH1F(varValues[0]+"_"+part, varValues[0]+"_"+part, 1001, 0., 1.001) ) # Histograms [3] for each variable name [0]. Loop appends and deletes entry each time.; range works for FW moments, need to update for others

    for event in jettree:
        for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
            varValues[3].Fill(varValues[2].GetValue()) # Fill histograms [3] with appropriate leaves [2]
    
    # Now we draw histograms onto canvases:
    
    for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        varValues[3].SetTitle(part)
        varValues[1].Add(varValues[3]) # Draw histogram [3]

        temphist = varValues[3].Clone("temphist")
        temphist.SetTitle(str(boost)+"GeV")
        # varValues[3].SetTitle(str(boost)+"GeV")
        if ( (boost > 10) and ( (boost % 15) == 0 ) ): # Fill Coarse Scan every 15 GeV, starting with 15 GeV (so 15, 30, 45, 60,...)
            partValues[3][var+part+"_Coarse_Scan"].Add(temphist) # Draw histogram [3]
            # partValues[3][var+part+"_Coarse_Scan"].Add(varValues[3]) # Draw histogram [3]

        if (partValues[0] - partValues[1]) <= boost <= (partValues[0] + partValues[1]): # If boost is near rest mass, fill Fine Scan
            partValues[3][var+part+"_Fine_Scan"].Add(temphist) # Draw histogram [3]    
            # partValues[3][var+part+"_Fine_Scan"].Add(varValues[3]) # Draw histogram [3]    
            
        del varValues[3] # Delete histogram [3] now that we are done
        del temphist

# End of particle loop
for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
    bpath = "pngs/boosts/"+var[:-1]+"/" # unique directory for each variable, trimming the underscore
    if not os.path.exists(bpath): os.makedirs(bpath) # If directory doesn't exist, create it

    cboost = ROOT.TCanvas(varValues[0])
    cboost.cd()
    varValues[1].Draw("PMC PLC")
    blegend = ROOT.gPad.BuildLegend()
    blegend.SetNColumns(2)
    ROOT.gPad.Update()
    cboost.SaveAs(bpath+varValues[0]+".png") # Save plot as png 
    cboost.Close() # Close canvas now that we are done


# open each scan canvas, fill with histo, close and delete everything
for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
    for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        ppath = "pngs/scans/"+var[:-1]+"/"
        if not os.path.exists(ppath): os.makedirs(ppath) # If directory doesn't exist, create it

        cpart1 = ROOT.TCanvas(var+part+"_Coarse_Scan")
        cpart1.cd()
        partValues[3][var+part+"_Coarse_Scan"].Draw("PMC PLC") 
        clegend1 = ROOT.gPad.BuildLegend()
        clegend1.SetNColumns(3)
        ROOT.gPad.Update()
        cpart1.SaveAs(ppath+var+part+"_Coarse_Scan.png") # Fill coarse canvas, close
        cpart1.Close()

        cpart2 = ROOT.TCanvas(var+part+"_Fine_Scan")
        cpart2.cd()
        partValues[3][var+part+"_Fine_Scan"].Draw("PMC PLC") 
        clegend2 = ROOT.gPad.BuildLegend()
        clegend2.SetNColumns(3)
        ROOT.gPad.Update()
        cpart2.SaveAs(ppath+var+part+"_Fine_Scan.png") # Fill fine canvas, close
        cpart2.Close()
    
    partValues[2].Close() # Close all of the root files 

#check how long the script took to run
runf = open("timeLog", "w") 
runf.write("Script took "+ str( (time.time() - startTime)/3600. ) + " hours to complete.")
runf.close