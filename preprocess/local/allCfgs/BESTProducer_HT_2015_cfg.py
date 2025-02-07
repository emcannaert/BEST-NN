from PhysicsTools.PatAlgos.tools.helpers  import getPatAlgosToolsTask
import FWCore.ParameterSet.Config as cms
process = cms.Process("analysis")
from Configuration.AlCa.GlobalTag import GlobalTag
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')
process.load('JetMETCorrections.Configuration.CorrectedJetProducers_cff')
process.load('JetMETCorrections.Configuration.CorrectedJetProducersDefault_cff')
process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
process.GlobalTag.globaltag = '106X_mcRun2_asymptotic_v17'
process.options = cms.untracked.PSet( allowUnscheduled = cms.untracked.bool(True) )
from PhysicsTools.PatUtils.tools.runMETCorrectionsAndUncertainties import runMetCorAndUncFromMiniAOD
################# JEC ################
corrLabels = ['L1FastJet','L2Relative','L3Absolute']
from PhysicsTools.PatAlgos.tools.jetTools import *
from RecoBTag.ONNXRuntime.pfDeepBoostedJet_cff import *
updateJetCollection(
 process,
 jetSource = cms.InputTag('slimmedJetsAK8'),
 labelName = 'AK8',
 jetCorrections = ('AK8PFPuppi', cms.vstring(corrLabels), 'None'), #previous corrections: 'L2Relative', 'L3Absolute', 'L2L3Residual'
 postfix = 'UpdatedJEC',
 printWarning = False
)
updateJetCollection(
 process,
 jetSource = cms.InputTag('slimmedJets'),
 labelName = 'AK4',
 jetCorrections = ('AK4PFchs', cms.vstring(corrLabels), 'None'),
 postfix = 'UpdatedJEC',
 printWarning = False
)  
################# Jet PU ID ################
from RecoJets.JetProducers.PileupJetID_cfi import pileupJetId
process.load("RecoJets.JetProducers.PileupJetID_cfi")
process.pileupJetIdUpdated = process.pileupJetId.clone( 
jets=cms.InputTag("selectedUpdatedPatJetsAK4UpdatedJEC"),  #should be the name of the post-JEC jet collection
inputIsCorrected=True,
applyJec=False,
vertexes=cms.InputTag("offlineSlimmedPrimaryVertices"),
)
process.patAlgosToolsTask.add(process.pileupJetIdUpdated)
updateJetCollection(    # running in unscheduled mode, need to manually update collection
 process,
 labelName = "PileupJetID",
 jetSource = cms.InputTag("selectedUpdatedPatJetsAK4UpdatedJEC"),
)
process.updatedPatJetsPileupJetID.userData.userInts.src = ["pileupJetIdUpdated:fullId"]
process.content = cms.EDAnalyzer("EventContentAnalyzer")
##############################################################################
process.leptonVeto = cms.EDFilter("leptonVeto",
 muonCollection= cms.InputTag("slimmedMuons"),
 electronCollection = cms.InputTag("slimmedElectrons"),
 metCollection = cms.InputTag("slimmedMETs"),
 tauCollection = cms.InputTag("slimmedTaus")
)
process.hadronFilter = cms.EDFilter("hadronFilter",
 year = cms.string("2015"),
 fatJetCollection = cms.InputTag("selectedUpdatedPatJetsAK8UpdatedJEC"),
 jetCollection = cms.InputTag("selectedUpdatedPatJetsPileupJetID"),
 bits = cms.InputTag("TriggerResults", "", "HLT"),
)
process.run = cms.EDProducer("BESTProducer",
 jetType = cms.string("HT"),
 genPartCollection = cms.string("prunedGenParticles"),
 fatJetCollection = cms.InputTag("selectedUpdatedPatJetsAK8UpdatedJEC"),
 jetCollection = cms.InputTag("selectedUpdatedPatJetsPileupJetID"),
 year = cms.string("2015"), #types: 2015,2016,2017,2018
 bits = cms.InputTag("TriggerResults", "", "HLT"),
 triggers = cms.string("HLT_PFHT900_v"),
)
process.source = cms.Source("PoolSource",
fileNames = cms.untracked.vstring(" /store/mc/RunIISummer20UL16MiniAODAPVv2/SuuToChiChiToHTHTToJets_MSuu-4000_MChi-1000_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/60000/EF8D3226-9665-FF42-8F73-20E1B71B2703.root"
)
)
process.TFileService = cms.Service("TFileService",fileName = cms.string("BESTInputs_HT_2015_output.root")
)
process.options = cms.untracked.PSet(
wantSummary = cms.untracked.bool(True),
)
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.p = cms.Path(  process.pileupJetIdUpdated * 
process.leptonVeto  * 
process.hadronFilter * 
  process.run)
process.patAlgosToolsTask = getPatAlgosToolsTask(process)
process.pathRunPatAlgos = cms.Path(process.patAlgosToolsTask)
