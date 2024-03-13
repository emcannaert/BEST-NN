import FWCore.ParameterSet.Config as cms

process = cms.Process("analysis")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring("/store/mc/RunIISummer20UL16MiniAODAPVv2/RSGluonToTT_M-1500_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/270000/796375D3-D7C1-AC49-B984-11A7BF201095.root")
)

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(5)
)

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(True),
)
#process.Tracer = cms.Service("Tracer")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.printTree = cms.EDAnalyzer("ParticleTreeDrawer",
   src = cms.InputTag("prunedGenParticles"),
   printP4 = cms.untracked.bool(False),
   printPtEtaPhi = cms.untracked.bool(False),
   printVertex = cms.untracked.bool(False),
   printStatus = cms.untracked.bool(True),
   printIndex = cms.untracked.bool(False),
   #status = cms.untracked.vint32(3),
)

process.p = cms.Path(
   process.printTree
)


