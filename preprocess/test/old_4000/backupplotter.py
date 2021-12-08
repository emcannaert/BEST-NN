import ROOT
import sys
import os
import time

######### Good lord fix the comments ########

startTime = time.time()

# cmslpc154

ROOT.gROOT.SetBatch(1) # This prevents any windows from being displayed

# Create the vector of boosts used in BESTProducer
boosts = [15,75,80,81,90,180]
# boosts = []
# iterMass = 1
# while iterMass <= 200: 
#     boosts.append(iterMass) # Add this mass to the vector, then increment by 1 GeV if any condition is true, or 5 GeV if none are true. 
#     if ( (iterMass < 15) or (80 <= iterMass < 95) or ( 165 <= iterMass < 180) ): 
#         iterMass += 1
#     else:
#         iterMass += 5


# dict = { "PARTICLES":{ "HH": [125., 10., tfile, cstring ,ccanvas ,fstring ,fcanvas ], OTHERPARTICLES...}, "VARIABLES": {"FoxWolfH1_":[bstring,canvas,leaf], "FW2":... , "OTHER VARS":... } }
# Create dictionary. Add variables and particles to the initial dictionary/list as desired
# initialPartDict = { "bb":[10., 4.]}
# initialPartDict = { "WW":[80.,6]}
initialPartDict = { "HH":[125., 12.], "WW":[80.,6]}
# initialPartDict = {"HH":[115.,20.], "WW":[65.,15.], "ZZ":[80.,7.], "tt":[155.,11.], "bb":[70.,10.], "QCD":[60.,20]}
# initialVarDict = {variable:[None,None] for variable in ["FoxWolfH1_", "FoxWolfH2_"] }
initialVarDict = {variable:{} for variable in ["FoxWolfH1_", "FoxWolfH2_"] }
# initialVarDict= {variable:[None,None,None] for variable in ["FoxWolfH1_", "FoxWolfH2_", "FoxWolfH3_", "FoxWolfH4_", "isotropy_", "sphericity_", "aplanarity_", 
#                                                             "thrust_", "nJets_", "jet12_mass_", "jet23_mass_", "jet13_mass_", "jet1234_mass_", "jet12_CosTheta_",
#                                                             "jet23_CosTheta_", "jet13_CosTheta_", "jet1234_CosTheta_", "jet12_DeltaCosTheta_", "jet13_DeltaCosTheta_", 
#                                                             "jet23_DeltaCosTheta_", "asymmetry_"] }

                                            
# jetAK8_mass

Dictionary = { "PARTICLES":initialPartDict , "VARIABLES":initialVarDict } # Will append root things to the particle dict, and will replace the values in the variable dict several times

# Iterate over dictionary values containing rest mass [0] and mass scan range [1] for each particle, and add root things:
for part, partValues in Dictionary["PARTICLES"].items():
    partValues.append(ROOT.TFile.Open(part+"_BESTInputs.root")) # TFile [2] for each particle
    partValues.append( {} ) # Empty dictionary [3] to fill
    for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        partValues[3][var+part+"_Coarse_Scan"] = ROOT.THStack(var+part+"_Coarse_Scan",var+part+"_Coarse_Scan") # Coarse Scan String [3][key] for each particle 
        partValues[3][var+part+"_Fine_Scan"] = ROOT.THStack(var+part+"_Fine_Scan",var+part+"_Fine_Scan")    # Fine Scan String [3][key] for each particle # Fine Scan Canvas [6] for each particle
        # partValues[3][var+part+"_Coarse_Scan"] = ROOT.TCanvas(var+part+"_Coarse_Scan") # Coarse Scan String [3][key] for each particle 
        # partValues[3][var+part+"_Fine_Scan"] = [ROOT.TCanvas(var+part+"_Fine_Scan"), "PMC PLC"]    # Fine Scan String [3][key] for each particle # Fine Scan Canvas [6] for each particle

# print(Dictionary)
# Now dictionary has the form: {particle:[restmass, restmass range, TFile object, coarse TCanvas object, coarse TH1F object, fine TCanvas object, fine TH1F object]}

# Two sets of plots: 
# Boost set, which plots all particles for each boost (everything needs to be reset each loop)
# Coarse/Fine set, which plots many boosts for each particle (setup before loop, fill through whole loop, draw after loop)
coarseDrawOpt = "PMC PLC"
for boost in boosts: # Iterate over boosts
    print(boost)
    # boostDrawOpt = "PMC PLC" # Set boost canvas draw option to automatically choose colors. Will update this string later to include "SAME"
    boostDrawOpt = "SAME" # Set boost canvas draw option to automatically choose colors. Will update this string later to include "SAME"

    # Create canvas for each variable for this boost
    for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        # Make this match above?
        # varValues[0] = var+str(boost)+"GeV" # String [0] for identifying corresponding histogram for this variable for this boost
        # varValues[1] = ROOT.TCanvas(boostString) # Create canvas [1] for each variable for this boost by name [0]
        # boostCanvas = varValues[1] # Canvas for this variable for this boost (fill with 6 particles)
        # Change name to varString?
        boostString = var+str(boost)+"GeV" # Name of the histogram for this variable for this boost
        boostCanvas = ROOT.THStack(boostString,boostString) # Canvas for this variable for this boost (fill with 6 particles)
        # boostCanvas = ROOT.TCanvas(boostString) # Canvas for this variable for this boost (fill with 6 particles)


        for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
            print(part)
            # Assign dictionary entries to variables for readability:
            restMass    = partValues[0] # Rest mass of each particle
            massRange   = partValues[1] # Range from rest mass to examine for fine scan
            jettree     = partValues[2].Get("run/jetTree") # Get jet tree from the TFile for this particle. Root file structure is part+"_BESTInputs.root/run/jetTree/[leaves]"

            coarseString    = var+part+"_Coarse_Scan" # 
            coarseCanvas    = partValues[3][coarseString]
            fineString      = var+part+"_Fine_Scan"
            fineCanvas      = partValues[3][fineString]
            # fineCanvas      = partValues[3][fineString][0]
            # fineDrawOpt     = partValues[3][fineString][1]
            
            # if not ( ((partValues[0] - partValues[1]) <= boost <= (partValues[0] + partValues[1])) or ((boost > 20) and ((boost % 15) == 0)) ): continue # Only take coarse/fine scans
            # if not ( (partValues[0] - partValues[1]) <= boost <= (partValues[0] + partValues[1]) ): continue # Only take fine scans         

            # varValues[2] = jettree.GetLeaf(varValues[0]) # Leaves [2] where variable data are stored by name [0]
            #1-condition to inverse
            # varValues[2] = jettree.Draw(varValues[0], "(jetAK8_mass>85)*(jetAK8_mass<95)") # Leaves [2] where variable data are stored by name [0]
            
            # boostCanvas.cd()
            # jettree.Draw(boostString, "", boostDrawOpt) # Draw histogram
            jettree.Draw(boostString, "") # Draw histogram

            # varValues.append( ROOT.TH1F(varValues[0]+"_"+part, varValues[0]+"_"+part, 1001, 0., 1.001) ) # Histograms [3] for each variable name [0]. Loop appends and deletes entry each time.; range works for FW moments, need to update for others


            # hist = ROOT.gROOT.FindObject("htemp").Clone("hist")
            # varValues[part]=(ROOT.gROOT.FindObject("htemp").Clone(part))
            # varValues[boostString+"_"+part]=hist
            # htemp = varValues[boostString+"_"+part]
            htemp = ROOT.gROOT.FindObject("htemp")
            htemp.SetTitle(part)
            # varValues[boostString+"_"+part].SetTitle(part)
            # boostCanvas.Add(varValues[boostString+"_"+part])
            # boostCanvas.Add(ROOT.gROOT.FindObject("htemp").Clone(part))
            boostCanvas.Add(htemp.Clone(part))
            
            # temphist = ROOT.gROOT.FindObject("htemp").Clone("temphist")
            # temphist = varValues[boostString+"_"+part].Clone("temphist")
            # temphist.SetTitle(str(boost)+"GeV")
            htemp.SetTitle(str(boost)+"GeV")

            if ( (boost > 10) and ( (boost % 15) == 0 ) ): # Fill Coarse Scan every 15 GeV, starting with 15 GeV (so 15, 30, 45, 60,...)
                # coarseCanvas.cd()
                # hist.Draw(coarseDrawOpt)
                # jettree.Draw(boostString, "", coarseDrawOpt) # Draw histogram
                # jettree.Draw(boostString, "", "SAME PLC") # Draw histogram
                coarseCanvas.Add(htemp.Clone(str(boost)+"GeV"))
                # partValues[3][var+part+"_Coarse_Scan"].Add(varValues[3]) # Draw histogram [3]

            if (restMass - massRange) <= boost <= (restMass + massRange): # If boost is near rest mass, fill Fine Scan
                # fineCanvas.cd()         
                # if ( (fineCanvas.IsDrawn()) and (fineDrawOpt == "PMC PLC") ): fineDrawOpt = "SAME PMC PLC" # If there is already something drawn on the       
                # hist.Draw(fineDrawOpt)
                # jettree.Draw(boostString, "", fineDrawOpt) # Draw histogram
                fineCanvas.Add(htemp.Clone(str(boost)+"GeV"))
                # jettree.Draw(boostString, "", "SAME") # Draw histogram

                # partValues[3][var+part+"_Fine_Scan"].Add(varValues[3]) # Draw histogram [3]    
            # del htemp
            # del hist
            # del temphist
            # retrieve htemp, update title?
            # varValues.append( ROOT.TH1F(varValues[0]+"_"+part, varValues[0]+"_"+part, 1001, 0., 1.001) ) # Histograms [3] for each variable name [0]. Loop appends and deletes entry each time.; range works for FW moments, need to update for others
            # del hist
            # if boostDrawOpt == "": boostDrawOpt = "SAME" # After the first particle loop, draw boost histos on the same canvas
        # End particle loop

        # Save boost plot
        # bpath = "pngs/boosts/"+var[:-1]+"/" # unique directory for each variable, trimming the underscore
        # if not os.path.exists(bpath): os.makedirs(bpath) # If directory doesn't exist, create it
        # boostCanvas.cd()
        # boostLegend = ROOT.gPad.BuildLegend()
        # boostLegend.SetNColumns(2)
        # ROOT.gPad.Update()
        # boostCanvas.SaveAs(bpath+boostString+".png") # Save plot as png 
        # boostCanvas.Close() # Close canvas now that we are done

        bpath = "pngs/boosts/"+var[:-1]+"/" # unique directory for each variable, trimming the underscore
        if not os.path.exists(bpath): os.makedirs(bpath) # If directory doesn't exist, create it

        cboost = ROOT.TCanvas(boostString)
        cboost.cd()
        boostCanvas.Draw("PMC PLC NOSTACK")
        blegend = ROOT.gPad.BuildLegend()
        blegend.SetNColumns(2)
        ROOT.gPad.Update()
        cboost.SaveAs(bpath+boostString+".png") # Save plot as png 
        cboost.Close() # Close canvas now that we are done

    # End variable loop
    # if boost == 20: coarseDrawOpt = "SAME PMC PLC" # First coarse histo is at 15GeV, so update draw options before the next coarse histo at 30GeV
# End boost loop


# open each scan canvas, fill with histo, close and delete everything
for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
    for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        ppath = "pngs/scans/"+var[:-1]+"/"
        if not os.path.exists(ppath): os.makedirs(ppath) # If directory doesn't exist, create it

        cpart1 = ROOT.TCanvas(var+part+"_Coarse_Scan")
        cpart1.cd()
        partValues[3][var+part+"_Coarse_Scan"].Draw("PMC PLC NOSTACK") 
        clegend1 = ROOT.gPad.BuildLegend()
        clegend1.SetNColumns(3)
        ROOT.gPad.Update()
        cpart1.SaveAs(ppath+var+part+"_Coarse_Scan.png") # Fill coarse canvas, close
        cpart1.Close()

        cpart2 = ROOT.TCanvas(var+part+"_Fine_Scan")
        cpart2.cd()
        partValues[3][var+part+"_Fine_Scan"].Draw("PMC PLC NOSTACK") 
        clegend2 = ROOT.gPad.BuildLegend()
        clegend2.SetNColumns(3)
        ROOT.gPad.Update()
        cpart2.SaveAs(ppath+var+part+"_Fine_Scan.png") # Fill fine canvas, close
        cpart2.Close()
    
    partValues[2].Close() # Close all of the root files 

# open each scan canvas, fill with histo, close and delete everything
# for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
#     for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
#         coarseString    = var+part+"_Coarse_Scan" # 
#         coarseCanvas    = partValues[3][coarseString]
#         fineString      = var+part+"_Fine_Scan"
#         fineCanvas      = partValues[3][fineString][0]

#         ppath = "pngs/scans/"+var[:-1]+"/"
#         if not os.path.exists(ppath): os.makedirs(ppath) # If directory doesn't exist, create it
        
#         coarseCanvas.cd()
#         coarseLegend = ROOT.gPad.BuildLegend()
#         coarseLegend.SetNColumns(3)
#         # ROOT.gPad.SetDrawOption("PMC")
#         ROOT.gPad.Update()
#         coarseCanvas.SaveAs(ppath+coarseString+".png") # Fill coarse canvas, close
#         coarseCanvas.Close()

#         fineCanvas.cd()
#         fineLegend = ROOT.gPad.BuildLegend()
#         fineLegend.SetNColumns(3)
#         # ROOT.gPad.SetDrawOption("PLC")
#         ROOT.gPad.Update()
#         fineCanvas.SaveAs(ppath+fineString+".png") # Fill fine canvas, close
#         fineCanvas.Close()
    
#     partValues[2].Close() # Close all of the root files 



    #     # Create canvas for each variable for this boost
    # for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
    #     # Make this match above
    #     varValues[0] = var+str(boost)+"GeV" # String [0] for identifying corresponding histogram for this variable for this boost
    #     varValues[1] = ROOT.TCanvas(varValues[0]) # Create canvas [1] for each variable for this boost by name [0]

    # # All canvases need to be made before this point. bcanvas each boost loop, coarse/fine before all loops
    # for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
    #     # Assign dictionary entries to variables for readability:
    #     restMass    = partValues[0] # Rest mass of each particle
    #     massRange   = partValues[1] # Range from rest mass to examine for fine scan
    #     jettree     = partValues[2].Get("run/jetTree") # Get jet tree from the TFile for this particle. Root file structure is part+"_BESTInputs.root/run/jetTree/[leaves]"

    #     coarseString    = var+part+"_Coarse_Scan" # 
    #     coarseCanvas    = partValues[3][coarseString]
    #     fineString      = var+part+"_Fine_Scan"
    #     fineCanvas      = partValues[3][fineString][0]
    #     fineDrawOpt     = partValues[3][fineString][1]
        
    #     # Add reabable vars, fill things. dont mess with htemp until after test. 


    #     # if not ( ((partValues[0] - partValues[1]) <= boost <= (partValues[0] + partValues[1])) or ((boost > 20) and ((boost % 15) == 0)) ): continue # Only take coarse/fine scans
    #     # if not ( (partValues[0] - partValues[1]) <= boost <= (partValues[0] + partValues[1]) ): continue # Only take fine scans         
    #     print(part)

    #     for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
    #         # Assign dictionary entries to variables for readability:
    #         boostString = varValues[0] # Name of the histogram for this variable for this boost
    #         boostCanvas = varValues[1] # Canvas for this variable for this boost (fill with 6 particles)

    #         bpath = "pngs/boosts/"+var[:-1]+"/" # unique directory for each variable, trimming the underscore
    #         if not os.path.exists(bpath): os.makedirs(bpath) # If directory doesn't exist, create it

    #         # varValues[2] = jettree.GetLeaf(varValues[0]) # Leaves [2] where variable data are stored by name [0]
    #         #1-condition to inverse
    #         # varValues[2] = jettree.Draw(varValues[0], "(jetAK8_mass>85)*(jetAK8_mass<95)") # Leaves [2] where variable data are stored by name [0]
            
    #         boostCanvas.cd()
    #         jettree.Draw(boostString, "", boostDrawOpt) # Draw histogram


    #         if ( (boost > 10) and ( (boost % 15) == 0 ) ): # Fill Coarse Scan every 15 GeV, starting with 15 GeV (so 15, 30, 45, 60,...)
    #             coarseCanvas.cd()
    #             jettree.Draw(boostString, "", coarseDrawOpt) # Draw histogram
    #             # partValues[3][var+part+"_Coarse_Scan"].Add(varValues[3]) # Draw histogram [3]

    #         if (partValues[0] - partValues[1]) <= boost <= (partValues[0] + partValues[1]): # If boost is near rest mass, fill Fine Scan
    #             fineCanvas.cd()              
    #             if ( (fineCanvas.IsDrawn()) and (fineDrawOpt == "PMC PLC") ): fineDrawOpt = "SAME PMC PLC" # If there is already something drawn on the       
    #             jettree.Draw(boostString, "", fineDrawOpt) # Draw histogram

    #             # partValues[3][var+part+"_Fine_Scan"].Add(varValues[3]) # Draw histogram [3]    
                
    #         # retrieve htemp, update title?
    #         # varValues.append( ROOT.TH1F(varValues[0]+"_"+part, varValues[0]+"_"+part, 1001, 0., 1.001) ) # Histograms [3] for each variable name [0]. Loop appends and deletes entry each time.; range works for FW moments, need to update for others
    #     if boostDrawOpt == "PMC PLC": boostDrawOpt = "SAME PMC PLC" # After the first particle loop, draw boost histos on the same canvas
    # if boost == 20: coarseDrawOpt = "SAME PMC PLC" # First coarse histo is at 15GeV, so update draw options before the next coarse histo at 30GeV

    
        # for event in jettree:
        #     for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        #         varValues[3].Fill(varValues[2].GetValue()) # Fill histograms [3] with appropriate leaves [2]
        
        # Now we draw histograms onto canvases:
        
        # for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
        #     varValues[3].SetTitle(part)
        #     varValues[1].Add(varValues[3]) # Draw histogram [3]

        #     temphist = varValues[3].Clone("temphist")
        #     temphist.SetTitle(str(boost)+"GeV")
        #     # varValues[3].SetTitle(str(boost)+"GeV")
        #     if ( (boost > 10) and ( (boost % 15) == 0 ) ): # Fill Coarse Scan every 15 GeV, starting with 15 GeV (so 15, 30, 45, 60,...)
        #         partValues[3][var+part+"_Coarse_Scan"].Add(temphist) # Draw histogram [3]
        #         # partValues[3][var+part+"_Coarse_Scan"].Add(varValues[3]) # Draw histogram [3]

        #     if (partValues[0] - partValues[1]) <= boost <= (partValues[0] + partValues[1]): # If boost is near rest mass, fill Fine Scan
        #         partValues[3][var+part+"_Fine_Scan"].Add(temphist) # Draw histogram [3]    
        #         # partValues[3][var+part+"_Fine_Scan"].Add(varValues[3]) # Draw histogram [3]    
                
        #     del varValues[3] # Delete histogram [3] now that we are done
        #     del temphist

    # End of particle loop
    # for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
    #     bpath = "pngs/boosts/"+var[:-1]+"/" # unique directory for each variable, trimming the underscore
    #     if not os.path.exists(bpath): os.makedirs(bpath) # If directory doesn't exist, create it

    #     cboost = ROOT.TCanvas(varValues[0])
    #     cboost.cd()
    #     varValues[1].Draw("PMC PLC")
    #     blegend = ROOT.gPad.BuildLegend()
    #     blegend.SetNColumns(2)
    #     ROOT.gPad.Update()
    #     cboost.SaveAs(bpath+varValues[0]+".png") # Save plot as png 
    #     cboost.Close() # Close canvas now that we are done

# End of boost loop

# # open each scan canvas, fill with histo, close and delete everything
# for part, partValues in Dictionary["PARTICLES"].items(): # Iterate over each particle
#     for var, varValues in Dictionary["VARIABLES"].items(): # This iterates over the variables in the root file
#         ppath = "pngs/scans/"+var[:-1]+"/"
#         if not os.path.exists(ppath): os.makedirs(ppath) # If directory doesn't exist, create it

#         cpart1 = ROOT.TCanvas(var+part+"_Coarse_Scan")
#         cpart1.cd()
#         partValues[3][var+part+"_Coarse_Scan"].Draw("PMC PLC") 
#         clegend1 = ROOT.gPad.BuildLegend()
#         clegend1.SetNColumns(3)
#         ROOT.gPad.Update()
#         cpart1.SaveAs(ppath+var+part+"_Coarse_Scan.png") # Fill coarse canvas, close
#         cpart1.Close()

#         cpart2 = ROOT.TCanvas(var+part+"_Fine_Scan")
#         cpart2.cd()
#         partValues[3][var+part+"_Fine_Scan"].Draw("PMC PLC") 
#         clegend2 = ROOT.gPad.BuildLegend()
#         clegend2.SetNColumns(3)
#         ROOT.gPad.Update()
#         cpart2.SaveAs(ppath+var+part+"_Fine_Scan.png") # Fill fine canvas, close
#         cpart2.Close()
    
#     partValues[2].Close() # Close all of the root files 

#check how long the script took to run
runf = open("timeLog", "w") 
# runf.write("Script took "+ str( (time.time() - startTime)/3600. ) + " hours to complete.")
runf.write("Script took "+ str( time.time() - startTime ) + " seconds to complete.")
runf.close