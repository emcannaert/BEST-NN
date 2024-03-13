import FWCore.ParameterSet.Config as cms

process = cms.Process("analysis")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring("/store/mc/RunIISummer20UL17MiniAODv2/TTbar01Jets_TypeIHeavyN-El_4L_LO_MN120_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v3/2820000/AC5C1CB5-F9CF-BA47-A991-4786C4FDBA9B.root") #"/store/mc/RunIISummer20UL18MiniAODv2/TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/0237BF1C-8B33-F149-A182-564D3602B2D8.ro")
)

process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(50)
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


