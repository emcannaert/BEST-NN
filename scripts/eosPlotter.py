import sys
import os
import time
import numpy
import subprocess
from datasetDictionary import datasetDict # Import the dictionary of sample files

# os.system("eval `scramv1 runtime -sh`") # This is cmsenv
import ROOT

#sort keys to loop in order
# title for nostack plots part->var+part
#using slim jets to cut in run_part.py, collection seems to be different than what we are using (slimmed)?????

# fix bins, steal ak8soft?
# cmslpc138

#### INITIALIZE STUFF ####
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

##### CREATE BOOST VECTOR AND DICTIONARIES #####

# Create the vector of boosts used in BESTProducer:
# boosts = [15,75,80,81,90,180]
# boosts = [1]
# boosts = []
# boosts.append("ak8_SoftDrop")
# boosts.append("ak8")
# iterMass = 50
# while iterMass <= 400: 
#     boosts.append(iterMass) # Add this mass to the vector, then increment by 1 GeV if any condition is true, or 5 GeV if none are true. 
#     iterMass += 50
#     # if ( (110 <= iterMass < 160) or (180 <= iterMass < 220) ):  iterMass += 1
#     # elif                (iterMass < 400):                       iterMass += 5
#     # else:                                                       iterMass += 100

# Create particle dictionary: { particle:[rest mass, mass range, empty root file dictionary], ...}
# partDict = { "bb":[10., 4.]}
# partDict = { "HH":[125., 12.], "WW":[80.,6]}
# partDict = {"HH":[120.,5.,{}], "WW":[80.,10.,{}], "ZZ":[90.,10.,{}], "tt":[170.,10.,{}], "bb":[5.,30.,{}], "QCD":[100.,10.,{}]}
partDict = { "HH":[{},{}], "WW":[{},{}], "ZZ":[{},{}], "tt":[{},{}], "bb":[{},{}], "QCD":[{},{}] }

# Create boost variable dictionary: { variable:empty dict }
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

# listworks = ['jet12_DeltaCosTheta_150GeV', 'FoxWolfH2_Higgs', 'jet13_DeltaCosTheta_150GeV', 'nJets_50GeV', 'thrust_W', 'FoxWolfH2_ak8', 'asymmetry_150GeV', 'thrust_Z', 'jet23_CosTheta_Bottom', 'asymmetry_400GeV', 'nJets_200GeV', 'jet13_CosTheta_Top', 'FoxWolfH3_350GeV', 'asymmetry_Bottom', 'FoxWolfH3_300GeV', 'jet1234_CosTheta_200GeV', 'nJets_300GeV', 'thrust_250GeV', 'FoxWolfH4_300GeV', 'FoxWolfH1_ak8_SoftDrop', 'bDisc1', 'bDisc2', 'FoxWolfH4_400GeV', 'thrust_Higgs', 'jet12_DeltaCosTheta_Z', 'jet12_DeltaCosTheta_W', 'aplanarity_ak8_SoftDrop', 'jet23_mass_Top', 'jet12_mass_50GeV', 'jet1234_CosTheta_250GeV', 'jet13_CosTheta_ak8_SoftDrop', 'jet12_CosTheta_ak8_SoftDrop', 'FoxWolfH2_100GeV', 'FoxWolfH2_150GeV', 'jet12_DeltaCosTheta_ak8_SoftDrop', 'jet23_DeltaCosTheta_ak8', 'jet12_CosTheta_Higgs', 'jet23_DeltaCosTheta_100GeV', 'jet13_DeltaCosTheta_W', 'jetAK8_Tau21', 'jet13_DeltaCosTheta_Z', 'jet12_DeltaCosTheta_ak8', 'FoxWolfH1_300GeV', 'jet1234_CosTheta_150GeV', 'jet1234_mass_200GeV', 'jet1234_mass_ak8_SoftDrop', 'jet12_DeltaCosTheta_100GeV', 'jet12_CosTheta_Top', 'thrust_200GeV', 'jet12_DeltaCosTheta_Top', 'FoxWolfH3_Top', 'sphericity_300GeV', 'nJets_100GeV', 'jet23_CosTheta_Top', 'jet23_mass_W', 'FoxWolfH4_50GeV', 'asymmetry_300GeV', 'aplanarity_W', 'bDiscSubJet_Max', 'bDiscSubJet_Max', 'jetAK8_deepAK8MD_dnn_Largest', 'FoxWolfH3_100GeV', 'FoxWolfH4_Higgs', 'aplanarity_Z', 'sphericity_250GeV', 'jet13_CosTheta_Z', 'jetAK8_mass', 'jet13_CosTheta_W', 'asymmetry_200GeV', 'FoxWolfH1_200GeV', 'thrust_400GeV', 'jet1234_mass_100GeV', 'jet12_CosTheta_50GeV', 'FoxWolfH3_ak8', 'jet13_mass_200GeV', 'FoxWolfH4_350GeV', 'jet13_mass_ak8_SoftDrop', 'jet13_DeltaCosTheta_Higgs', 'jet12_mass_300GeV', 'FoxWolfH1_ak8', 'nJets_Z', 'nJets_W', 'jet12_DeltaCosTheta_50GeV', 'FoxWolfH1_100GeV', 'jet23_mass_50GeV', 'jet1234_mass_Higgs', 'thrust_300GeV', 'sphericity_W', 'jet12_CosTheta_ak8', 'jet1234_mass_150GeV', 'sphericity_Z', 'jet23_DeltaCosTheta_ak8_SoftDrop', 'jet13_CosTheta_300GeV', 'aplanarity_300GeV', 'nJets_ak8', 'jet12_mass_Higgs', 'jet12_CosTheta_200GeV', 'bDisc2_probb', 'jet1234_CosTheta_400GeV', 'jet12_CosTheta_300GeV', 'jet13_DeltaCosTheta_350GeV', 'thrust_100GeV', 'aplanarity_Top', 'FoxWolfH1_400GeV', 'jet12_mass_400GeV', 'thrust_150GeV', 'jet12_CosTheta_Bottom', 'jet1234_CosTheta_300GeV', 'jet23_mass_300GeV', 'jet1234_CosTheta_350GeV', 'FoxWolfH4_200GeV', 'jet23_mass_Bottom', 'sphericity_ak8_SoftDrop', 'jet12_DeltaCosTheta_400GeV', 'nJets_150GeV', 'jet12_mass_Bottom', 'FoxWolfH1_Top', 'jet12_DeltaCosTheta_300GeV', 'jet13_mass_ak8', 'jet13_mass_Higgs', 'jet12_DeltaCosTheta_200GeV', 'jet23_CosTheta_200GeV', 'FoxWolfH3_50GeV', 'sphericity_200GeV', 'jet23_DeltaCosTheta_400GeV', 'jetAK8_Tau32', 'jet12_CosTheta_350GeV', 'sphericity_100GeV', 'FoxWolfH2_300GeV', 'aplanarity_400GeV', 'jet13_CosTheta_400GeV', 'nJets_ak8_SoftDrop', 'FoxWolfH1_150GeV', 'jet1234_CosTheta_Bottom', 'jet23_DeltaCosTheta_150GeV', 'nJets_Top', 'aplanarity_Bottom', 'jet13_CosTheta_Bottom', 'jet23_CosTheta_Higgs', 'jet12_mass_250GeV', 'FoxWolfH3_400GeV', 'jet13_mass_100GeV', 'jet13_DeltaCosTheta_Top', 'jet23_mass_100GeV', 'FoxWolfH2_250GeV', 'nJets_Bottom', 'nJets', 'jet13_DeltaCosTheta_ak8_SoftDrop', 'jet12_DeltaCosTheta_Higgs', 'FoxWolfH3_150GeV', 'jet12_mass_W', 'jet12_mass_Z', 'jet13_DeltaCosTheta_200GeV', 'sphericity_400GeV', 'thrust_ak8', 'jet12_CosTheta_250GeV', 'jet13_CosTheta_250GeV', 'aplanarity_250GeV', 'jet23_mass_Z', 'jetAK8_charge', 'asymmetry_50GeV', 'jet12_mass_350GeV', 'jet23_mass_Higgs', 'jet12_mass_100GeV', 'jet13_mass_250GeV', 'jet1234_mass_Top', 'jet12_mass_ak8_SoftDrop', 'nJets_400GeV', 'jet12_CosTheta_Z', 'jet12_CosTheta_W', 'FoxWolfH4_Bottom', 'asymmetry_ak8', 'jet12_mass_150GeV', 'bDisc2_probbb', 'jet23_DeltaCosTheta_250GeV', 'jet1234_mass_250GeV', 'jet1234_CosTheta_ak8_SoftDrop', 'jet1234_mass_W', 'sphericity_Higgs', 'jet1234_mass_Z', 'jet13_CosTheta_350GeV', 'aplanarity_350GeV', 'thrust_Top', 'sphericity_Top', 'aplanarity_100GeV', 'jet13_CosTheta_100GeV', 'FoxWolfH1_Higgs', 'jetAK8_SoftDropMass', 'jet23_mass_ak8', 'jet1234_CosTheta_Z', 'jet13_mass_Z', 'jet13_mass_W', 'jet1234_CosTheta_W', 'FoxWolfH4_ak8_SoftDrop', 'asymmetry_W', 'asymmetry_Z', 'sphericity_150GeV', 'jet13_mass_Top', 'asymmetry_Top', 'aplanarity_150GeV', 'FoxWolfH3_250GeV', 'jet13_CosTheta_150GeV', 'FoxWolfH3_Higgs', 'jet23_DeltaCosTheta_200GeV', 'asymmetry_Higgs', 'bDiscSubJet_Max_index', 'bDiscSubJet_Max_index', 'jetAK8_deepAK8_dnn_Largest', 'jet1234_CosTheta_Top', 'jet23_CosTheta_100GeV', 'jet13_mass_150GeV', 'jet23_mass_350GeV', 'jet1234_CosTheta_100GeV', 'jet12_DeltaCosTheta_350GeV', 'jet1234_CosTheta_50GeV', 'jet13_DeltaCosTheta_ak8', 'FoxWolfH3_Z', 'FoxWolfH3_W', 'FoxWolfH1_Z', 'FoxWolfH4_250GeV', 'jet23_CosTheta_W', 'jet23_CosTheta_Z', 'bDisc', 'FoxWolfH1_W', 'FoxWolfH3_200GeV', 'jet13_mass_400GeV', 'jet23_mass_ak8_SoftDrop', 'jet13_mass_300GeV', 'jet23_DeltaCosTheta_50GeV', 'bDisc1_probbb', 'nSecondaryVertices', 'jet12_CosTheta_150GeV', 'jet23_CosTheta_250GeV', 'FoxWolfH4_Z', 'FoxWolfH4_W', 'jet13_mass_350GeV', 'jet12_mass_ak8', 'jet23_mass_200GeV', 'jet13_DeltaCosTheta_50GeV', 'jet12_CosTheta_100GeV', 'FoxWolfH1_Bottom', 'thrust_Bottom', 'jet1234_mass_ak8', 'FoxWolfH1_350GeV', 'sphericity_ak8', 'sphericity_Bottom', 'nJets_Higgs', 'aplanarity_50GeV', 'jet13_CosTheta_50GeV', 'FoxWolfH2_200GeV', 'jet23_DeltaCosTheta_Bottom', 'thrust_350GeV', 'jet23_mass_150GeV', 'jet1234_CosTheta_ak8', 'jet13_DeltaCosTheta_Bottom', 'FoxWolfH4_150GeV', 'jet13_DeltaCosTheta_400GeV', 'jet13_DeltaCosTheta_100GeV', 'jet12_DeltaCosTheta_Bottom', 'FoxWolfH1_50GeV', 'bDisc_probb', 'jet23_CosTheta_150GeV', 'jetAK8_deepAK8MD_rawZ', 'FoxWolfH2_50GeV', 'nJets_250GeV', 'jetAK8_pt', 'jet13_mass_Bottom', 'nJets_350GeV', 'FoxWolfH3_ak8_SoftDrop', 'FoxWolfH3_Bottom', 'FoxWolfH2_W', 'jet12_mass_Top', 'bDisc_probbb', 'FoxWolfH2_Z', 'FoxWolfH4_100GeV', 'jet23_CosTheta_400GeV', 'jet12_mass_200GeV', 'asymmetry_100GeV', 'jet23_CosTheta_300GeV', 'FoxWolfH2_ak8_SoftDrop', 'FoxWolfH2_Top', 'FoxWolfH2_350GeV', 'jet13_mass_50GeV', 'jetAK8_Tau4', 'jetAK8_Tau3', 'jetAK8_Tau2', 'jetAK8_Tau1', 'FoxWolfH2_Bottom', 'FoxWolfH4_ak8', 'jet1234_mass_300GeV', 'thrust_50GeV', 'jet13_DeltaCosTheta_250GeV', 'FoxWolfH1_250GeV', 'jet23_mass_400GeV', 'jet23_CosTheta_ak8_SoftDrop', 'asymmetry_ak8_SoftDrop', 'jet23_mass_250GeV', 'jet13_CosTheta_ak8', 'FoxWolfH4_Top', 'sphericity_50GeV', 'jet23_CosTheta_350GeV', 'aplanarity_ak8', 'jetAK8_deepAK8_rawB', 'jetAK8_deepAK8_rawC', 'jetAK8_deepAK8_rawH', 'jet13_CosTheta_200GeV', 'jetAK8_deepAK8_rawL', 'aplanarity_200GeV', 'jetAK8_deepAK8_rawW', 'jetAK8_deepAK8_rawT', 'jetAK8_deepAK8_rawZ', 'jet23_DeltaCosTheta_Higgs', 'jet13_DeltaCosTheta_300GeV', 'FoxWolfH2_400GeV', 'jet1234_mass_Bottom', 'jet12_CosTheta_400GeV', 'jet1234_mass_50GeV', 'thrust_ak8_SoftDrop', 'jet23_CosTheta_50GeV', 'jet1234_mass_350GeV', 'jet23_CosTheta_ak8', 'sphericity_350GeV', 'jet23_DeltaCosTheta_Top', 'bDisc1_probb', 'asymmetry_350GeV', 'jet12_DeltaCosTheta_250GeV', 'jet1234_mass_400GeV', 'jetAK8_deepAK8MD_rawC', 'jetAK8_deepAK8MD_rawB', 'jetAK8_deepAK8MD_rawH', 'jet1234_CosTheta_Higgs', 'jetAK8_deepAK8MD_rawL', 'jet23_DeltaCosTheta_300GeV', 'jetAK8_deepAK8MD_rawT', 'jetAK8_deepAK8MD_rawW', 'jet23_DeltaCosTheta_W', 'aplanarity_Higgs', 'jet23_DeltaCosTheta_Z', 'jet23_DeltaCosTheta_350GeV', 'jet13_CosTheta_Higgs', 'asymmetry_250GeV']
# listfails = ['TopFrame_PF_candidate_energy', 'WFrame_PF_candidate_pz', 'WFrame_PF_candidate_px', 'WFrame_PF_candidate_py', 'jet12_DeltaCosTheta_150GeV', 'FoxWolfH2_Higgs', 'ak8Frame_PF_candidate_pz', 'jet13_DeltaCosTheta_150GeV', '300GeVFrame_jet_py', '50GeVFrame_PF_candidate_energy', 'nJets_50GeV', 'thrust_W', 'FoxWolfH2_ak8', 'asymmetry_150GeV', '50GeVFrame_PF_candidate_py', 'isotropy_ak8', 'thrust_Z', 'jet23_CosTheta_Bottom', 'asymmetry_400GeV', 'nJets_200GeV', 'jet13_CosTheta_Top', 'FoxWolfH3_350GeV', 'asymmetry_Bottom', 'FoxWolfH3_300GeV', 'LabFrame_PF_candidate_isNeutralHadron', 'jet1234_CosTheta_200GeV', 'nJets_300GeV', 'thrust_250GeV', 'FoxWolfH4_300GeV', 'FoxWolfH1_ak8_SoftDrop', 'bDisc1', 'bDisc2', 'FoxWolfH4_400GeV', 'thrust_Higgs', 'BottomFrame_PF_candidate_py', 'BottomFrame_PF_candidate_px', 'BottomFrame_PF_candidate_pz', 'PUPPI_Weights', 'jet12_DeltaCosTheta_Z', '100GeVFrame_jet_energy', 'jet12_DeltaCosTheta_W', 'aplanarity_ak8_SoftDrop', 'jet23_mass_Top', 'jet12_mass_50GeV', 'jet1234_CosTheta_250GeV', 'jet13_CosTheta_ak8_SoftDrop', 'LabFrame_PF_candidate_logpT', 'ak8Frame_jet_energy', 'jet12_CosTheta_ak8_SoftDrop', '200GeVFrame_PF_candidate_energy', 'FoxWolfH2_100GeV', 'HiggsFrame_PF_candidate_py', 'HiggsFrame_PF_candidate_px', 'HiggsFrame_PF_candidate_pz', 'FoxWolfH2_150GeV', '400GeVFrame_jet_energy', 'jet12_DeltaCosTheta_ak8_SoftDrop', 'ak8Frame_jet_py', 'jet23_DeltaCosTheta_ak8', 'jet12_CosTheta_Higgs', 'jet23_DeltaCosTheta_100GeV', 'jet13_DeltaCosTheta_W', 'jetAK8_Tau21', '200GeVFrame_jet_energy', 'jet13_DeltaCosTheta_Z', 'TopFrame_jet_px', 'TopFrame_jet_py', 'TopFrame_jet_pz', 'HiggsFrame_PF_candidate_energy', 'jet12_DeltaCosTheta_ak8', 'ZFrame_PF_candidate_px', 'FoxWolfH1_300GeV', 'jet1234_CosTheta_150GeV', 'jet1234_mass_200GeV', 'jet1234_mass_ak8_SoftDrop', 'jet12_DeltaCosTheta_100GeV', 'jet12_CosTheta_Top', 'thrust_200GeV', 'jet12_DeltaCosTheta_Top', 'SV_pt', '250GeVFrame_jet_py', '250GeVFrame_jet_px', 'LabFrame_PF_candidate_deltaR', '250GeVFrame_jet_pz', 'FoxWolfH3_Top', '200GeVFrame_jet_px', '200GeVFrame_jet_py', '200GeVFrame_jet_pz', 'sphericity_300GeV', 'nJets_100GeV', 'jet23_CosTheta_Top', '300GeVFrame_PF_candidate_pz', '300GeVFrame_PF_candidate_py', '300GeVFrame_PF_candidate_px', 'jet23_mass_W', 'isotropy_ak8_SoftDrop', 'FoxWolfH4_50GeV', 'asymmetry_300GeV', 'aplanarity_W', 'bDiscSubJet_Max', 'jetAK8_deepAK8MD_dnn_Largest', 'FoxWolfH3_100GeV', 'FoxWolfH4_Higgs', 'aplanarity_Z', 'sphericity_250GeV', 'jet13_CosTheta_Z', '50GeVFrame_jet_energy', 'jetAK8_mass', 'jet13_CosTheta_W', 'asymmetry_200GeV', '300GeVFrame_jet_energy', 'FoxWolfH1_200GeV', 'thrust_400GeV', 'jet1234_mass_100GeV', 'jet12_CosTheta_50GeV', 'BottomFrame_jet_energy', 'FoxWolfH3_ak8', 'jetAK8_phi', 'jet13_mass_200GeV', 'isotropy_400GeV', 'FoxWolfH4_350GeV', 'jet13_mass_ak8_SoftDrop', 'jet13_DeltaCosTheta_Higgs', 'HiggsFrame_jet_py', '50GeVFrame_PF_candidate_pz', 'HiggsFrame_jet_pz', 'jet12_mass_300GeV', '50GeVFrame_PF_candidate_px', 'FoxWolfH1_ak8', 'isotropy_350GeV', 'nJets_Z', 'LabFrame_PF_candidate_pz', 'LabFrame_PF_candidate_px', 'nJets_W', 'jet12_DeltaCosTheta_50GeV', '350GeVFrame_jet_energy', 'FoxWolfH1_100GeV', 'jet23_mass_50GeV', 'jet1234_mass_Higgs', 'thrust_300GeV', 'sphericity_W', 'jet12_CosTheta_ak8', 'jet1234_mass_150GeV', 'sphericity_Z', 'SV_nTracks', 'jet23_DeltaCosTheta_ak8_SoftDrop', 'LabFrame_PF_candidate_pdgId', 'jet13_CosTheta_300GeV', 'aplanarity_300GeV', 'nJets_ak8', '300GeVFrame_PF_candidate_energy', 'jet12_mass_Higgs', '200GeVFrame_PF_candidate_pz', 'jet12_CosTheta_200GeV', 'ak8_SoftDropFrame_PF_candidate_py', 'bDisc2_probb', 'ak8_SoftDropFrame_PF_candidate_pz', 'jet1234_CosTheta_400GeV', 'ZFrame_jet_py', 'jet12_CosTheta_300GeV', 'ZFrame_jet_pz', 'jet13_DeltaCosTheta_350GeV', 'thrust_100GeV', '400GeVFrame_jet_py', 'aplanarity_Top', 'FoxWolfH1_400GeV', 'ak8_SoftDropFrame_jet_energy', 'jet12_mass_400GeV', 'BottomFrame_jet_py', 'BottomFrame_jet_px', 'BottomFrame_jet_pz', 'WFrame_jet_energy', 'thrust_150GeV', 'WFrame_PF_candidate_energy', 'ak8Frame_jet_pz', 'jet12_CosTheta_Bottom', 'jet1234_CosTheta_300GeV', 'jet23_mass_300GeV', 'jet1234_CosTheta_350GeV', 'FoxWolfH4_200GeV', 'ak8_SoftDropFrame_PF_candidate_energy', 'jet23_mass_Bottom', 'sphericity_ak8_SoftDrop', '100GeVFrame_jet_py', '100GeVFrame_jet_px', '200GeVFrame_PF_candidate_px', '100GeVFrame_jet_pz', '400GeVFrame_jet_pz', '400GeVFrame_jet_px', 'jet12_DeltaCosTheta_400GeV', 'nJets_150GeV', 'jet12_mass_Bottom', 'FoxWolfH1_Top', 'jet12_DeltaCosTheta_300GeV', 'jet13_mass_ak8', 'jet13_mass_Higgs', 'ZFrame_PF_candidate_py', 'jet12_DeltaCosTheta_200GeV', 'ZFrame_PF_candidate_pz', 'jet23_CosTheta_200GeV', 'FoxWolfH3_50GeV', 'sphericity_200GeV', 'jet23_DeltaCosTheta_400GeV', 'jetAK8_Tau32', 'HiggsFrame_jet_px', 'jet12_CosTheta_350GeV', 'sphericity_100GeV', 'FoxWolfH2_300GeV', 'aplanarity_400GeV', 'jet13_CosTheta_400GeV', 'nJets_ak8_SoftDrop', 'BottomFrame_PF_candidate_energy', 'FoxWolfH1_150GeV', 'jet1234_CosTheta_Bottom', 'jet23_DeltaCosTheta_150GeV', 'LabFrame_PF_candidate_abspdgId', 'nJets_Top', 'aplanarity_Bottom', 'jet13_CosTheta_Bottom', 'jet23_CosTheta_Higgs', 'jet12_mass_250GeV', 'FoxWolfH3_400GeV', 'jet13_mass_100GeV', 'jet13_DeltaCosTheta_Top', 'jet23_mass_100GeV', 'isotropy_300GeV', 'ZFrame_PF_candidate_energy', '250GeVFrame_PF_candidate_py', '250GeVFrame_PF_candidate_px', '250GeVFrame_PF_candidate_pz', 'FoxWolfH2_250GeV', 'nJets_Bottom', 'LabFrame_PF_candidate_py', 'nJets', 'jet13_DeltaCosTheta_ak8_SoftDrop', 'jet12_DeltaCosTheta_Higgs', 'FoxWolfH3_150GeV', 'isotropy_Higgs', 'jet12_mass_W', 'jet12_mass_Z', 'jet13_DeltaCosTheta_200GeV', 'sphericity_400GeV', 'thrust_ak8', 'jet12_CosTheta_250GeV', 'jet13_CosTheta_250GeV', 'aplanarity_250GeV', 'jet23_mass_Z', 'jetAK8_charge', '100GeVFrame_PF_candidate_energy', 'asymmetry_50GeV', 'jet12_mass_350GeV', 'jet23_mass_Higgs', 'LabFrame_PF_candidate_logEnergy', 'ak8Frame_PF_candidate_energy', 'jet12_mass_100GeV', 'jet13_mass_250GeV', 'jet1234_mass_Top', 'LabFrame_PF_candidate_logEnergyRatio', '200GeVFrame_PF_candidate_py', 'jet12_mass_ak8_SoftDrop', 'nJets_400GeV', 'jet12_CosTheta_Z', 'jet12_CosTheta_W', 'isotropy_200GeV', 'SV_mass', 'FoxWolfH4_Bottom', 'asymmetry_ak8', '350GeVFrame_PF_candidate_px', '350GeVFrame_PF_candidate_py', '350GeVFrame_PF_candidate_pz', '350GeVFrame_PF_candidate_energy', 'jet12_mass_150GeV', 'ak8_SoftDropFrame_PF_candidate_px', 'SV_chi2', 'bDisc2_probbb', 'jet23_DeltaCosTheta_250GeV', 'jet1234_mass_250GeV', 'jet1234_CosTheta_ak8_SoftDrop', 'jet1234_mass_W', 'sphericity_Higgs', 'jet1234_mass_Z', 'jet13_CosTheta_350GeV', 'aplanarity_350GeV', 'thrust_Top', 'sphericity_Top', 'aplanarity_100GeV', 'jet13_CosTheta_100GeV', 'FoxWolfH1_Higgs', 'jetAK8_SoftDropMass', 'jet23_mass_ak8', 'jet1234_CosTheta_Z', 'jet13_mass_Z', 'jet13_mass_W', 'jet1234_CosTheta_W', 'FoxWolfH4_ak8_SoftDrop', 'LabFrame_PF_candidate_energy', 'asymmetry_W', 'asymmetry_Z', 'sphericity_150GeV', 'jet13_mass_Top', 'WFrame_jet_px', 'SV_eta', 'asymmetry_Top', 'aplanarity_150GeV', 'FoxWolfH3_250GeV', 'jet13_CosTheta_150GeV', 'FoxWolfH3_Higgs', 'jet23_DeltaCosTheta_200GeV', 'asymmetry_Higgs', 'bDiscSubJet_Max_index', 'jetAK8_deepAK8_dnn_Largest', 'jet1234_CosTheta_Top', 'LabFrame_PF_candidate_isPhoton', 'jet23_CosTheta_100GeV', 'jet13_mass_150GeV', 'jet23_mass_350GeV', 'ak8_SoftDropFrame_jet_pz', 'ak8_SoftDropFrame_jet_px', 'ak8_SoftDropFrame_jet_py', 'jet1234_CosTheta_100GeV', 'TopFrame_PF_candidate_pz', '50GeVFrame_jet_px', '50GeVFrame_jet_py', '50GeVFrame_jet_pz', 'HiggsFrame_jet_energy', 'TopFrame_PF_candidate_px', 'jet12_DeltaCosTheta_350GeV', 'ak8Frame_PF_candidate_py', 'ak8Frame_PF_candidate_px', 'jet1234_CosTheta_50GeV', '250GeVFrame_jet_energy', 'jet13_DeltaCosTheta_ak8', 'FoxWolfH3_Z', 'LabFrame_PF_candidate_deltaPhi', 'FoxWolfH3_W', 'ZFrame_jet_energy', 'FoxWolfH1_Z', 'FoxWolfH4_250GeV', 'jet23_CosTheta_W', 'jet23_CosTheta_Z', 'bDisc', 'FoxWolfH1_W', 'FoxWolfH3_200GeV', 'jet13_mass_400GeV', 'jet23_mass_ak8_SoftDrop', 'jet13_mass_300GeV', 'LabFrame_PF_candidate_isChargedHadron', 'jet23_DeltaCosTheta_50GeV', 'bDisc1_probbb', 'nSecondaryVertices', 'jet12_CosTheta_150GeV', 'jet23_CosTheta_250GeV', 'TopFrame_jet_energy', '150GeVFrame_PF_candidate_pz', '150GeVFrame_PF_candidate_px', '150GeVFrame_PF_candidate_py', 'FoxWolfH4_Z', 'FoxWolfH4_W', 'ak8Frame_jet_px', 'jet13_mass_350GeV', 'LabFrame_PF_candidate_logpTRatio', 'jet12_mass_ak8', 'jet23_mass_200GeV', 'jetAK8_eta', 'jet13_DeltaCosTheta_50GeV', 'LabFrame_PF_candidate_deltaEta', 'jet12_CosTheta_100GeV', 'isotropy_250GeV', 'WFrame_jet_pz', 'FoxWolfH1_Bottom', '150GeVFrame_PF_candidate_energy', 'thrust_Bottom', 'jet1234_mass_ak8', 'FoxWolfH1_350GeV', 'sphericity_ak8', 'sphericity_Bottom', 'isotropy_Bottom', 'TopFrame_PF_candidate_py', 'isotropy_100GeV', 'nJets_Higgs', 'isotropy_Top', 'LabFrame_PF_candidate_charge', 'aplanarity_50GeV', 'jet13_CosTheta_50GeV', 'FoxWolfH2_200GeV', 'jet23_DeltaCosTheta_Bottom', 'thrust_350GeV', 'jet23_mass_150GeV', 'jet1234_CosTheta_ak8', 'jet13_DeltaCosTheta_Bottom', 'SV_Ndof', 'FoxWolfH4_150GeV', 'LabFrame_PF_candidate_isElectron', 'jet13_DeltaCosTheta_400GeV', 'jet13_DeltaCosTheta_100GeV', 'jet12_DeltaCosTheta_Bottom', 'FoxWolfH1_50GeV', 'bDisc_probb', '300GeVFrame_jet_pz', 'jet23_CosTheta_150GeV', '300GeVFrame_jet_px', 'jetAK8_deepAK8MD_rawZ', 'isotropy_Z', '150GeVFrame_jet_energy', 'FoxWolfH2_50GeV', 'isotropy_W', 'nJets_250GeV', 'jetAK8_pt', 'jet13_mass_Bottom', 'nJets_350GeV', 'FoxWolfH3_ak8_SoftDrop', 'FoxWolfH3_Bottom', 'FoxWolfH2_W', 'jet12_mass_Top', 'bDisc_probbb', 'FoxWolfH2_Z', 'SV_phi', 'FoxWolfH4_100GeV', 'jet23_CosTheta_400GeV', 'jet12_mass_200GeV', 'asymmetry_100GeV', '150GeVFrame_jet_px', '150GeVFrame_jet_py', '150GeVFrame_jet_pz', 'jet23_CosTheta_300GeV', 'FoxWolfH2_ak8_SoftDrop', 'FoxWolfH2_Top', 'FoxWolfH2_350GeV', 'jet13_mass_50GeV', 'jetAK8_Tau4', 'jetAK8_Tau3', 'jetAK8_Tau2', 'jetAK8_Tau1', 'FoxWolfH2_Bottom', 'FoxWolfH4_ak8', 'jet1234_mass_300GeV', 'thrust_50GeV', 'jet13_DeltaCosTheta_250GeV', 'FoxWolfH1_250GeV', 'jet23_mass_400GeV', 'jet23_CosTheta_ak8_SoftDrop', 'asymmetry_ak8_SoftDrop', 'jet23_mass_250GeV', 'jet13_CosTheta_ak8', 'FoxWolfH4_Top', 'sphericity_50GeV', 'jet23_CosTheta_350GeV', '250GeVFrame_PF_candidate_energy', 'aplanarity_ak8', 'ZFrame_jet_px', 'jetAK8_deepAK8_rawB', 'jetAK8_deepAK8_rawC', 'jetAK8_deepAK8_rawH', 'jet13_CosTheta_200GeV', 'WFrame_jet_py', 'jetAK8_deepAK8_rawL', 'aplanarity_200GeV', '100GeVFrame_PF_candidate_py', '100GeVFrame_PF_candidate_px', '100GeVFrame_PF_candidate_pz', 'jetAK8_deepAK8_rawW', 'jetAK8_deepAK8_rawT', 'LabFrame_PF_candidate_isMuon', 'jetAK8_deepAK8_rawZ', 'jet23_DeltaCosTheta_Higgs', 'isotropy_150GeV', 'jet13_DeltaCosTheta_300GeV', 'FoxWolfH2_400GeV', 'jet1234_mass_Bottom', 'jet12_CosTheta_400GeV', 'jet1234_mass_50GeV', '400GeVFrame_PF_candidate_px', '400GeVFrame_PF_candidate_py', '400GeVFrame_PF_candidate_pz', 'thrust_ak8_SoftDrop', 'jet23_CosTheta_50GeV', '400GeVFrame_PF_candidate_energy', 'jet1234_mass_350GeV', 'jet23_CosTheta_ak8', 'sphericity_350GeV', 'jet23_DeltaCosTheta_Top', 'bDisc1_probb', 'asymmetry_350GeV', 'jet12_DeltaCosTheta_250GeV', '350GeVFrame_jet_pz', 'isotropy_50GeV', '350GeVFrame_jet_px', '350GeVFrame_jet_py', 'jet1234_mass_400GeV', 'jetAK8_deepAK8MD_rawC', 'jetAK8_deepAK8MD_rawB', 'jetAK8_deepAK8MD_rawH', 'jet1234_CosTheta_Higgs', 'jetAK8_deepAK8MD_rawL', 'jet23_DeltaCosTheta_300GeV', 'jetAK8_deepAK8MD_rawT', 'jetAK8_deepAK8MD_rawW', 'jet23_DeltaCosTheta_W', 'aplanarity_Higgs', 'jet23_DeltaCosTheta_Z', 'jet23_DeltaCosTheta_350GeV', 'jet13_CosTheta_Higgs', 'asymmetry_250GeV']
# listdiff =  ['TopFrame_PF_candidate_energy', 'WFrame_PF_candidate_pz', 'WFrame_PF_candidate_px', 'WFrame_PF_candidate_py', 'ak8Frame_PF_candidate_pz', '300GeVFrame_jet_py', '50GeVFrame_PF_candidate_energy', '50GeVFrame_PF_candidate_py', 'isotropy_ak8', 'LabFrame_PF_candidate_isNeutralHadron', 'BottomFrame_PF_candidate_py', 'BottomFrame_PF_candidate_px', 'BottomFrame_PF_candidate_pz', 'PUPPI_Weights', '100GeVFrame_jet_energy', 'LabFrame_PF_candidate_logpT', 'ak8Frame_jet_energy', '200GeVFrame_PF_candidate_energy', 'HiggsFrame_PF_candidate_py', 'HiggsFrame_PF_candidate_px', 'HiggsFrame_PF_candidate_pz', '400GeVFrame_jet_energy', 'ak8Frame_jet_py', '200GeVFrame_jet_energy', 'TopFrame_jet_px', 'TopFrame_jet_py', 'TopFrame_jet_pz', 'HiggsFrame_PF_candidate_energy', 'ZFrame_PF_candidate_px', 'SV_pt', '250GeVFrame_jet_py', '250GeVFrame_jet_px', 'LabFrame_PF_candidate_deltaR', '250GeVFrame_jet_pz', '200GeVFrame_jet_px', '200GeVFrame_jet_py', '200GeVFrame_jet_pz', '300GeVFrame_PF_candidate_pz', '300GeVFrame_PF_candidate_py', '300GeVFrame_PF_candidate_px', 'isotropy_ak8_SoftDrop', '50GeVFrame_jet_energy', '300GeVFrame_jet_energy', 'BottomFrame_jet_energy', 'jetAK8_phi', 'isotropy_400GeV', 'HiggsFrame_jet_py', '50GeVFrame_PF_candidate_pz', 'HiggsFrame_jet_pz', '50GeVFrame_PF_candidate_px', 'isotropy_350GeV', 'LabFrame_PF_candidate_pz', 'LabFrame_PF_candidate_px', '350GeVFrame_jet_energy', 'SV_nTracks', 'LabFrame_PF_candidate_pdgId', '300GeVFrame_PF_candidate_energy', '200GeVFrame_PF_candidate_pz', 'ak8_SoftDropFrame_PF_candidate_py', 'ak8_SoftDropFrame_PF_candidate_pz', 'ZFrame_jet_py', 'ZFrame_jet_pz', '400GeVFrame_jet_py', 'ak8_SoftDropFrame_jet_energy', 'BottomFrame_jet_py', 'BottomFrame_jet_px', 'BottomFrame_jet_pz', 'WFrame_jet_energy', 'WFrame_PF_candidate_energy', 'ak8Frame_jet_pz', 'ak8_SoftDropFrame_PF_candidate_energy', '100GeVFrame_jet_py', '100GeVFrame_jet_px', '200GeVFrame_PF_candidate_px', '100GeVFrame_jet_pz', '400GeVFrame_jet_pz', '400GeVFrame_jet_px', 'ZFrame_PF_candidate_py', 'ZFrame_PF_candidate_pz', 'HiggsFrame_jet_px', 'BottomFrame_PF_candidate_energy', 'LabFrame_PF_candidate_abspdgId', 'isotropy_300GeV', 'ZFrame_PF_candidate_energy', '250GeVFrame_PF_candidate_py', '250GeVFrame_PF_candidate_px', '250GeVFrame_PF_candidate_pz', 'LabFrame_PF_candidate_py', 'isotropy_Higgs', '100GeVFrame_PF_candidate_energy', 'LabFrame_PF_candidate_logEnergy', 'ak8Frame_PF_candidate_energy', 'LabFrame_PF_candidate_logEnergyRatio', '200GeVFrame_PF_candidate_py', 'isotropy_200GeV', 'SV_mass', '350GeVFrame_PF_candidate_px', '350GeVFrame_PF_candidate_py', '350GeVFrame_PF_candidate_pz', '350GeVFrame_PF_candidate_energy', 'ak8_SoftDropFrame_PF_candidate_px', 'SV_chi2', 'LabFrame_PF_candidate_energy', 'WFrame_jet_px', 'SV_eta', 'LabFrame_PF_candidate_isPhoton', 'ak8_SoftDropFrame_jet_pz', 'ak8_SoftDropFrame_jet_px', 'ak8_SoftDropFrame_jet_py', 'TopFrame_PF_candidate_pz', '50GeVFrame_jet_px', '50GeVFrame_jet_py', '50GeVFrame_jet_pz', 'HiggsFrame_jet_energy', 'TopFrame_PF_candidate_px', 'ak8Frame_PF_candidate_py', 'ak8Frame_PF_candidate_px', '250GeVFrame_jet_energy', 'LabFrame_PF_candidate_deltaPhi', 'ZFrame_jet_energy', 'LabFrame_PF_candidate_isChargedHadron', 'TopFrame_jet_energy', '150GeVFrame_PF_candidate_pz', '150GeVFrame_PF_candidate_px', '150GeVFrame_PF_candidate_py', 'ak8Frame_jet_px', 'LabFrame_PF_candidate_logpTRatio', 'jetAK8_eta', 'LabFrame_PF_candidate_deltaEta', 'isotropy_250GeV', 'WFrame_jet_pz', '150GeVFrame_PF_candidate_energy', 'isotropy_Bottom', 'TopFrame_PF_candidate_py', 'isotropy_100GeV', 'isotropy_Top', 'LabFrame_PF_candidate_charge', 'SV_Ndof', 'LabFrame_PF_candidate_isElectron', '300GeVFrame_jet_pz', '300GeVFrame_jet_px', 'isotropy_Z', '150GeVFrame_jet_energy', 'isotropy_W', 'SV_phi', '150GeVFrame_jet_px', '150GeVFrame_jet_py', '150GeVFrame_jet_pz', '250GeVFrame_PF_candidate_energy', 'ZFrame_jet_px', 'WFrame_jet_py', '100GeVFrame_PF_candidate_py', '100GeVFrame_PF_candidate_px', '100GeVFrame_PF_candidate_pz', 'LabFrame_PF_candidate_isMuon', 'isotropy_150GeV', '400GeVFrame_PF_candidate_px', '400GeVFrame_PF_candidate_py', '400GeVFrame_PF_candidate_pz', '400GeVFrame_PF_candidate_energy', '350GeVFrame_jet_pz', 'isotropy_50GeV', '350GeVFrame_jet_px', '350GeVFrame_jet_py']
# newlist = []
# for f in listfails:
#     flg = False
#     for w in listworks:
#         if w == f: flg = True
#     if flg == False: newlist.append(f)

# print(listdiff)
                                
print("Checking directory for root files...")
# partDict["WW"][2]["WW_M-3000"] = ROOT.TFile.Open("root://cmsxrootd.fnal.gov//store/user/msabbott/BulkGravToWWToWhadWhad_narrow_M-3000_TuneCP5_13TeV-madgraph-pythia/crab_GravitonWW_3000GeV_trees/211115_234227/0000/all.root")
#
# tfile = ROOT.TFile.Open("root://cmseos.fnal.gov//store/user/maabbott/ZprimeToBB_narrow_M-5500_TuneCP5_13TeV-madgraph-pythia8/crab_ZprimeBB_M_5500GeV_trees/211124_161815/0000/BESTInputs_10.root")
# ttree = tfile.Get("run/jetTree") 
# floats = []
# notfloats = []

# # for var in listfails:
# for var in listdiff:
#     tleaf = ttree.GetLeaf(var)
#     ttype = tleaf.GetTypeName()
#     print(ttype + ": " + var)
#     # notfloats.append(ttype + ": " + var)
#     # if ttype == "Float_t": floats.append(var)
#     # else: notfloats.append(ttype + ": " + var)
# notfloats.sort()

# # print("floats: " + str(floats))
# # print()
# print(notfloats)
# # newleaf = ttree.GetLeaf("nJets")
# # newtype = newleaf.GetTypeName()
# # print(newtype)
# # print(ttype)
# # wfile = ROOT.TFile.Open("root://cmsxrootd.fnal.gov//store/user/msabbott/BulkGravToWWToWhadWhad_narrow_M-3000_TuneCP5_13TeV-madgraph-pythia/crab_GravitonWW_3000GeV_trees/211115_234227/0000/")
# # print("opened all.root")

# """

for part, subDicts in partDict.items():
    if part == "tt" or part == "QCD":
        keys = datasetDict["mc"]["2017"][part].keys()
        keys.sort()
        subDicts.append(keys)
        # for key in datasetDict["mc"]["2017"][part].keys():
    else:
        keys = datasetDict["mc"]["2017"][part].keys()
        keys.sort(key=int)
        subDicts.append(keys)
# print(partDict)

# print(wfile)
# Use eosls to get list of BESTInputs
# Turn this into loop over datasetdict
# Timestamp is an issue. Know everything else. Can use .checkoutput to append timestamp onto path.
# eospath = "/store/user/msabbott/"
eosls = ["xrdfs", "root://cmseos.fnal.gov", "ls", "-u"] # Alias for eosls, broken up into pieces for subprocess 
for part, subDicts in partDict.items():
    chainDict =  subDicts[0]
    stackDict =  subDicts[1]
    massPoints = subDicts[2]
    for massPoint in massPoints:
        crabInfo = datasetDict["mc"]["2017"][part][massPoint]
    # for massPoint, crabInfo in datasetDict["mc"]["2017"][part].items(): # Iterate through dictionary by mass point (key) and [crabdir,dataset,file] (value)
        endIndex = crabInfo[1].find('Run') # Identify eos dir name
        eospath = ["/store/user/msabbott" + crabInfo[1][:endIndex] + "crab_" + crabInfo[0]] # Example: /store/user/msabbott/BulkGravToWWToWhadWhad_narrow_M-3000_TuneCP5_13TeV-madgraph-pythia/crab_GravitonWW_3000GeV_trees
        timeStamp = subprocess.check_output(eosls + eospath).split("\n") # This should give the timestamps, code works for only one timestamp (can edit)
        if len(timeStamp) > 2: # Check that there is only one timestamp. Final entry of list should always be empty string from the .split("\n")
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

# print(partDict) 
print("Plotting lab frame variables...")

# for part, partValues in partDict.items():
for labvar, binInfo in labFrameDict.items(): # This iterates over the lab frame variables in the root file
    finalStack = ROOT.THStack(labvar,labvar)
    
    # labPath = plotPath+"lab/"+labvar+"/"
    allPath = plotPath+"all/"+labvar+"/"
    if not os.path.exists(allPath): os.makedirs(allPath) # If directory doesn't exist, create it        

    for part, subDicts in partDict.items():
        chainDict =  subDicts[0]
        stack =      subDicts[1][labvar]
        massPoints = subDicts[2]

        indvPath = plotPath+"indvidual/"+labvar+"/"+part+"/"
        if not os.path.exists(indvPath): os.makedirs(indvPath) # If directory doesn't exist, create it        

        # hsum = ROOT.TH1F(labvar+"_"+part,labvar+"_"+part,binInfo)
        hsum = ROOT.TH1F(labvar+"_"+part,labvar+"_"+part,100,500.,4500.)

        ROOT.gStyle.SetPalette(ROOT.kCool) #109 kCool palette for indvidual plots
        for massPoint in massPoints:
            chain = chainDict[massPoint]
        # for massPoint, chain in chainDict.items():


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

        
        ROOT.gStyle.SetPalette(ROOT.kRainBow) # kRainBow palette for all together plots; bad for 2D data but fine here 
        # Now redraw, then plot all mass points separetely for this particle
        stack.Draw("PMC PLC HIST NOSTACK") # Draw all histos in Stack at once. PMC/PLC for auto colors, NOSTACK to keep histos separate
        legend = ROOT.gPad.BuildLegend(0.3,1.,1.) # Create legend at coords, top right of canvas
        legend.SetNColumns(7) # Set legend columns
        ROOT.gPad.Update() # Draw legend
        canvas.SaveAs(allPath+labvar+"_"+part+".png") # Save plot as png 
        canvas.Close() # Close canvas now that we are done
    
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