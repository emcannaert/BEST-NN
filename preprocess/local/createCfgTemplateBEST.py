#! /usr/bin/env python

import sys
import os
import pickle

def makeACfg(jetType, year, datafile):

   # need to change trigger for yhears
   trigger = ""
   if year == "2018" or year == "2017":
      trigger = "HLT_PFHT1050_v"
   elif year == "2015" or year == "2016":
      trigger = "HLT_PFHT900_v"

   output_dir = "allCfgs/"

   newCfg = open("%s/BESTProducer_%s_%s_cfg.py"%(output_dir,jetType,year),"w")

 
   newCfg.write("from PhysicsTools.PatAlgos.tools.helpers  import getPatAlgosToolsTask\n")
   newCfg.write("import FWCore.ParameterSet.Config as cms\n")

   newCfg.write('process = cms.Process("analysis")\n')
   newCfg.write("from Configuration.AlCa.GlobalTag import GlobalTag\n")  

   newCfg.write("process.load('Configuration.StandardSequences.Services_cff')\n")
   newCfg.write('process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")\n')
   newCfg.write("process.load('JetMETCorrections.Configuration.JetCorrectors_cff')\n")
   newCfg.write("process.load('JetMETCorrections.Configuration.CorrectedJetProducers_cff')\n")
   newCfg.write("process.load('JetMETCorrections.Configuration.CorrectedJetProducersDefault_cff')\n")
   newCfg.write('process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")\n')
   newCfg.write('process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")\n')

   newCfg.write('from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection\n')

   # global tags
   #NEW and what is in use: https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun2LegacyAnalysis

   if year == "2018":
      newCfg.write("process.GlobalTag.globaltag = '106X_upgrade2018_realistic_v16_L1v1'\n")  
   elif year == "2017":
      newCfg.write("process.GlobalTag.globaltag = '106X_mc2017_realistic_v10'\n")
   elif year == "2016":
      newCfg.write("process.GlobalTag.globaltag = '106X_mcRun2_asymptotic_v17'\n")
   elif year == "2015":
      newCfg.write("process.GlobalTag.globaltag = '106X_mcRun2_asymptotic_v17'\n")
   else: 
      print("Something wrong with sample and the global tag designation.")

   newCfg.write("process.options = cms.untracked.PSet( allowUnscheduled = cms.untracked.bool(True) )\n")

   newCfg.write("from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD\n")

   newCfg.write("################# JEC ################\n")
   newCfg.write("corrLabels = ['L1FastJet','L2Relative','L3Absolute']\n")
   newCfg.write("from PhysicsTools.PatAlgos.tools.jetTools import *\n")
   newCfg.write("from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import *\n")
   newCfg.write("updateJetCollection(\n")
   newCfg.write(" process,\n")
   newCfg.write(" jetSource = cms.InputTag('slimmedJetsAK8'),\n")
   newCfg.write(" labelName = 'AK8',\n")
   newCfg.write(" jetCorrections = ('AK8PFPuppi', cms.vstring(corrLabels), 'None'), #previous corrections: 'L2Relative', 'L3Absolute', 'L2L3Residual'\n")
   newCfg.write(" postfix = 'UpdatedJEC',\n")
   newCfg.write(" printWarning = False\n")
   newCfg.write(")\n")
   newCfg.write("updateJetCollection(\n")
   newCfg.write(" process,\n")
   newCfg.write(" jetSource = cms.InputTag('slimmedJets'),\n")
   newCfg.write(" labelName = 'AK4',\n")
   newCfg.write(" jetCorrections = ('AK4PFchs', cms.vstring(corrLabels), 'None'),\n")
   newCfg.write(" postfix = 'UpdatedJEC',\n")
   newCfg.write(" printWarning = False\n")
   newCfg.write(")  \n")


   newCfg.write("################# Jet PU ID ################\n")
   newCfg.write('from RecoJets.JetProducers.PileupJetID_cfi import pileupJetId\n')
   if year == "2016":
      newCfg.write('from RecoJets.JetProducers.PileupJetID_cfi import _chsalgos_106X_UL16   #   (or _chsalgos_106X_UL16APV for APV samples)\n')
   elif year == "2017":
      newCfg.write('from RecoJets.JetProducers.PileupJetID_cfi import _chsalgos_106X_UL17\n')
   elif year == "2018":
      newCfg.write('from RecoJets.JetProducers.PileupJetID_cfi import _chsalgos_106X_UL18\n')
   newCfg.write('process.load("RecoJets.JetProducers.PileupJetID_cfi")\n')
   newCfg.write('process.pileupJetIdUpdated = process.pileupJetId.clone( \n')
   newCfg.write('jets=cms.InputTag("selectedUpdatedPatJetsAK4UpdatedJEC"),  #should be the name of the post-JEC jet collection\n')
   newCfg.write('inputIsCorrected=True,\n')
   newCfg.write('applyJec=False,\n')
   newCfg.write('vertexes=cms.InputTag("offlineSlimmedPrimaryVertices"),\n')
   if year == "2016":
      newCfg.write('algos = cms.VPSet(_chsalgos_106X_UL16),\n')
   elif year == "2017":
      newCfg.write('algos = cms.VPSet(_chsalgos_106X_UL17),\n')
   elif year == "2018":
      newCfg.write('algos = cms.VPSet(_chsalgos_106X_UL18),\n')
   newCfg.write(')\n')


   newCfg.write('process.patAlgosToolsTask.add(process.pileupJetIdUpdated)\n')
   newCfg.write('updateJetCollection(    # running in unscheduled mode, need to manually update collection\n')
   newCfg.write(' process,\n')
   newCfg.write(' labelName = "PileupJetID",\n')
   newCfg.write(' jetSource = cms.InputTag("selectedUpdatedPatJetsAK4UpdatedJEC"),\n')
   newCfg.write(')\n')
   newCfg.write('process.updatedPatJetsPileupJetID.userData.userInts.src = ["pileupJetIdUpdated:fullId"]\n')
   newCfg.write('process.content = cms.EDAnalyzer("EventContentAnalyzer")\n')




   newCfg.write("##############################################################################\n")
   newCfg.write('process.leptonVeto = cms.EDFilter("leptonVeto",\n')
   newCfg.write(' muonCollection= cms.InputTag("slimmedMuons"),\n')
   newCfg.write(' electronCollection = cms.InputTag("slimmedElectrons"),\n')
   newCfg.write(' metCollection = cms.InputTag("slimmedMETs"),\n')
   newCfg.write(' tauCollection = cms.InputTag("slimmedTaus")\n')
   newCfg.write(")\n")



   newCfg.write('process.hadronFilter = cms.EDFilter("hadronFilter",\n')
   newCfg.write(' year = cms.string("%s"),\n'%year)
   newCfg.write(' fatJetCollection = cms.InputTag("selectedUpdatedPatJetsAK8UpdatedJEC"),\n')
   newCfg.write(' jetCollection = cms.InputTag("selectedUpdatedPatJetsPileupJetID"),\n')
   newCfg.write(' bits = cms.InputTag("TriggerResults", "", "HLT"),\n')
   #newCfg.write(' triggers = cms.string("%s"),\n'%trigger)
   newCfg.write(")\n")



   newCfg.write('process.run = cms.EDProducer("BESTProducer",\n')
   newCfg.write(' jetType = cms.string("%s"),\n'%jetType)
   newCfg.write(' genPartCollection = cms.string("prunedGenParticles"),\n')

   newCfg.write(' fatJetCollection = cms.InputTag("selectedUpdatedPatJetsAK8UpdatedJEC"),\n')
   newCfg.write(' jetCollection = cms.InputTag("selectedUpdatedPatJetsPileupJetID"),\n')
   newCfg.write(' year = cms.string("%s"), #types: 2015,2016,2017,2018\n'%year)
   newCfg.write(' bits = cms.InputTag("TriggerResults", "", "HLT"),\n')
   newCfg.write(' triggers = cms.string("%s"),\n'%trigger)
   newCfg.write(")\n")


   newCfg.write('process.source = cms.Source("PoolSource",\n')
   newCfg.write('fileNames = cms.untracked.vstring("%s"\n'%datafile)
   newCfg.write(")\n")
   newCfg.write(")\n")



   newCfg.write('process.TFileService = cms.Service("TFileService",fileName = cms.string("BESTInputs_%s_%s_output.root")\n'%(jetType,year))

   newCfg.write(")\n")

   newCfg.write("process.options = cms.untracked.PSet(\n")
   newCfg.write("wantSummary = cms.untracked.bool(True),\n")
   newCfg.write(")\n")

   newCfg.write('process.load("FWCore.MessageLogger.MessageLogger_cfi")\n')
   newCfg.write("process.MessageLogger.cerr.FwkReport.reportEvery = 1000\n")



   newCfg.write("process.p = cms.Path(  ")
   newCfg.write("process.pileupJetIdUpdated * \n")
   newCfg.write("process.leptonVeto  * \n")
   newCfg.write("process.hadronFilter * \n ")   
   newCfg.write(" process.run" )

   newCfg.write(")\n")
   newCfg.write("process.patAlgosToolsTask = getPatAlgosToolsTask(process)\n")
   newCfg.write("process.pathRunPatAlgos = cms.Path(process.patAlgosToolsTask)\n")

   newCfg.close()

def main():

   years = ["2015","2016","2017","2018"]


   # just one file per year per type of jet for testing purposes 
   datafiles = { '2015': { "WB": " /store/mc/RunIISummer20UL16MiniAODAPVv2/SuuToChiChiToWBWBToJets_MSuu-4000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/70000/4669F1D8-D68A-964D-BF3C-68384E82D1CA.root", 
                           "HT":" /store/mc/RunIISummer20UL16MiniAODAPVv2/SuuToChiChiToHTHTToJets_MSuu-4000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/60000/EF8D3226-9665-FF42-8F73-20E1B71B2703.root", 
                           "ZT":"/store/mc/RunIISummer20UL16MiniAODAPVv2/SuuToChiChiToZTZTToJets_MSuu-4000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/2550000/2D6D4CB0-0947-5742-B5BB-C6D1CDA57A4F.root", 
                           "QCD":"/store/mc/RunIISummer20UL16MiniAODAPVv2/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/120000/13041AC9-85AD-AF4C-A4FF-DA0CC3525903.root", 
                           "Top":"/store/mc/RunIISummer20UL16MiniAODAPVv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/120000/0343A84D-6182-D149-B7E1-2B824263CC61.root",
                            "ST":"/store/mc/RunIISummer20UL16MiniAODAPVv2/ST_t-channel_antitop_4f_InclusiveDecays_mtop1755_TuneCP5_13TeV-powheg-madspin-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v3/120000/0D27B966-7F95-1A40-B3E0-B401D0CAC86F.root",
                             "WJets":"/store/mc/RunIISummer20UL16MiniAODAPVv2/EWKWplus2Jets_WToQQ_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2/2830000/16688C77-5ED3-204A-8E29-AE508B46701B.root" },
               '2016': { "WB":"/store/mc/RunIISummer20UL16MiniAODv2/SuuToChiChiToWBWBToJets_MSuu-4000_MChi-1500_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v3/80000/32F31612-9557-7D40-9B51-CDFB67D4280C.root" , 
                         "HT":"/store/mc/RunIISummer20UL16MiniAODv2/SuuToChiChiToHTHTToJets_MSuu-8000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v1/2560000/E5F26A5B-8B35-5643-B69D-D801433EDEE2.root", 
                         "ZT":"/store/mc/RunIISummer20UL16MiniAODv2/SuuToChiChiToZTZTToJets_MSuu-4000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v1/60000/8D07D15B-AB1D-E348-9014-0BFB49566112.root", 
                         "QCD": "/store/mc/RunIISummer20UL16MiniAODv2/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraphMLM-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/0605C02A-A789-C548-8D73-9B7D0A4D5561.root", 
                         "Top": "/store/mc/RunIISummer20UL16MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v1/120000/0006FC43-9130-8E41-9330-98229A8C074E.root" ,
                         "ST":"/store/mc/RunIISummer20UL16MiniAODv2/ST_t-channel_top_4f_InclusiveDecays_mtop1755_TuneCP5_13TeV-powheg-madspin-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v3/120000/053E5FAA-5A87-CE45-B431-2E9C654302AC.root",
                         "WJets":"/store/mc/RunIISummer20UL16MiniAODv2/EWKWplus2Jets_WToQQ_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v1/2820000/02087E33-8EB9-0B4C-B904-40298EF293F8.root"},
               '2017': {"WB":"/store/mc/RunIISummer20UL17MiniAODv2/SuuToChiChiToWBWBToJets_MSuu-6000_MChi-2000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v2/2550000/B3091170-ED1B-5541-A637-49977AA8D808.root" , 
                        "HT":"/store/mc/RunIISummer20UL17MiniAODv2/SuuToChiChiToHTHTToJets_MSuu-4000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v3/50000/763A4729-98C0-8743-9FDE-D69F99D67F1C.root", 
                        "ZT":"/store/mc/RunIISummer20UL17MiniAODv2/SuuToChiChiToZTZTToJets_MSuu-8000_MChi-1500_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v3/50000/E27674C6-57D0-274D-A615-D0A5954D4657.root", 
                        "QCD":"/store/mc/RunIISummer20UL17MiniAODv2/QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v2/2530000/00725C6E-5FF7-A34B-A71E-759B53B7FBA4.root", 
                        "Top":"/store/mc/RunIISummer20UL17MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v2/110000/029813D5-62C1-A84A-8711-5EEA17D5C096.root",
                         "ST":"/store/mc/RunIISummer20UL17MiniAODv2/ST_t-channel_antitop_4f_InclusiveDecays_mtop1755_TuneCP5_13TeV-powheg-madspin-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v3/120000/1B43EE0D-67F5-1740-AAA4-5F0911AB26E5.root",
                         "WJets":"/store/mc/RunIISummer20UL16MiniAODAPVv2/EWKWplus2Jets_WToQQ_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v2/2830000/16688C77-5ED3-204A-8E29-AE508B46701B.root" },
               '2018': {"WB": "/store/mc/RunIISummer20UL18MiniAODv2/SuuToChiChiToWBWBToJets_MSuu-7000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2550000/0C2BCADD-4AE6-E845-8716-E45781824073.root", 
                        "HT":"/store/mc/RunIISummer20UL18MiniAODv2/SuuToChiChiToHTHTToJets_MSuu-5000_MChi-1500_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v3/50000/AAD2FFDB-710E-3D4F-A166-014A30CEFAD2.root", 
                        "ZT":"/store/mc/RunIISummer20UL18MiniAODv2/SuuToChiChiToZTZTToJets_MSuu-5000_MChi-2000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2560000/BBDBABFC-E1F4-0A4A-8F31-F304DB605706.root", 
                        "QCD":"/store/mc/RunIISummer20UL18MiniAODv2/QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2520000/070C2B0E-738E-224F-B3F5-0B0E6D807ABB.root", 
                        "Top":"/store/mc/RunIISummer20UL18MiniAODv2/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/00000/004EF875-ACBB-FE45-B86B-EAF83448CE62.root",
                         "ST":"/store/mc/RunIISummer20UL18MiniAODv2/ST_t-channel_antitop_4f_InclusiveDecays_mtop1755_TuneCP5_13TeV-powheg-madspin-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v3/120000/01E59DDE-18FF-3F4B-ADB7-4088888FCBD4.root",
                          "WJets":"/store/mc/RunIISummer20UL18MiniAODv2/EWKWplus2Jets_WToQQ_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2820000/09C0E70F-ED9A-1F46-B506-DCE9BE2F76E1.root" } }

   #samples = ["Suu_high_mass", "Suu_low_mass", "QCDMC", "TTbarMC"]
   jetTypes = ["WB", "HT", "ZT", "QCD", "Top", "ST", "WJets"]

   num_files = 0
   for year in years:
      for jetType in jetTypes:
         makeACfg(jetType, year, datafiles[year][jetType])  
         num_files+=1 
   print("Finished with %i files."%num_files )
   return


if __name__ == "__main__":
    main()
