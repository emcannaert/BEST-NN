#=========================================================================================
# run_tt.py ------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Samantha Abbott, Johan Bonilla, Sydney Ostrom, Reyer Band, -
#            Congqiao Li, Anna Benecke ---------------------------------------------------
#-----------------------------------------------------------------------------------------

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from Configuration.AlCa.GlobalTag import GlobalTag

GT = "106X_mc2017_realistic_v8"
process = cms.Process("run")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load("JetMETCorrections.Configuration.JetCorrectionServices_cff")
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, GT)

# process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20000))

process.source = cms.Source("PoolSource",
        # Replace root file below with the source file you want to use (overwritten by crab config files that call this run file)
        fileNames = cms.untracked.vstring(
        # "/store/mc/RunIISummer20UL16MiniAODv2/ZprimeToTT_M500_W5_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v1/280000/83459D51-7262-AE4D-9BC5-EDDE4114C688.root"
		# "/store/mc/RunIISummer20UL16MiniAODAPVv2/ZprimeToTT_M1000_W10_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/280000/CF5D5D25-0E5A-A446-A38F-2493C6AEB879.root"
        "/store/mc/RunIISummer20UL17MiniAODv2/ZprimeToTT_M4000_W40_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/240000/28EB1B0B-1962-3A44-8706-56B2364205F7.root"
        # "/store/mc/RunIISummer20UL16MiniAODv2/ZprimeToTT_M4500_W45_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v1/280000/5478AFB3-51C3-4D45-A431-46F0D34940F2.root"
                                         )
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#=========================================================================================
# Add Deep AK8 variables -----------------------------------------------------------------
#=========================================================================================

_btagDiscriminators = [ 
    'pfDeepFlavourJetTags:probb', 'pfDeepFlavourJetTags:probbb', 
    'pfDeepFlavourJetTags:problepb', 'pfDeepFlavourJetTags:probc',
    'pfDeepFlavourJetTags:probuds', 'pfDeepFlavourJetTags:probg'
    ]
updateJetCollection(
    process,
    labelName='SoftDropSubjetsPF',
    jetSource=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked", "SubJets"),
    jetCorrections=('AK4PFPuppi',
                    ['L2Relative', 'L3Absolute'], 'None'),
    btagDiscriminators=list(_btagDiscriminators),
    explicitJTA=True,  # needed for subjet b tagging
    svClustering=False,  # needed for subjet b tagging (IMPORTANT: Needs to be set to False to disable ghost-association which does not work with slimmed jets)
    fatJets=cms.InputTag('slimmedJetsAK8'),  # needed for subjet b tagging
    rParam=0.8,  # needed for subjet b tagging
    sortByPt=False, # Don't change order (would mess with subJetIdx for FatJets)
    postfix='AK8DF')

#updateJetCollection(
#   process,
#   jetSource = cms.InputTag('slimmedJetsAK8'),
#   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
#   svSource = cms.InputTag('slimmedSecondaryVertices'),
#   rParam = 0.8,
#   jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
#   btagDiscriminators = ['pfDeepFlavourJetTags:probb', 'pfDeepFlavourJetTags:probbb', 'pfDeepFlavourJetTags:problepb', 'pfDeepFlavourJetTags:probc',
#                         'pfDeepFlavourJetTags:probuds', 'pfDeepFlavourJetTags:probg'],
#    postfix = 'AK8DF',
#    printWarning = False # Making this false removes the "b tagging need to be run on uncorrected jets" warning, which would print for every job.
#)


#=========================================================================================
# Prepare and run producer ---------------------------------------------------------------
#=========================================================================================

# Apply a preselction
process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
                                        src = cms.InputTag('slimmedJetsAK8'),
                                        cut = cms.string('500.0 < pt && pt < 3500.0 && abs(eta) < 2.4'),
                                        filter = cms.bool(True)
)

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
                                    minNumber = cms.uint32(1),
                                    maxNumber = cms.uint32(99999),
                                    src = cms.InputTag('slimmedJetsAK8')
                                    #filter = cms.bool(True)
)

# Run the producer
process.run = cms.EDProducer('BESTProducer',
                             inputJetColl = cms.string('selectedAK8Jets'),
                             inputSubJetColl = cms.string('updatedPatJetsTransientCorrectedSoftDropSubjetsPFAK8DF'),
                             jetColl = cms.string('PUPPI'),                     
			                 jetType = cms.string("t"),
                             storeDaughters = cms.bool(True),
)
process.TFileService = cms.Service("TFileService", fileName = cms.string("tt_4000_BESTInputs.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("tt_ana_out.root"),
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_fixedGridRhoAll_*_*',
                                                                      'keep *_run_*_*'
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
)
process.outpath = cms.EndPath(process.out)

# Organize the running procedure
#process.tsk = cms.Task()
#for mod in process.producers_().itervalues():
#    process.tsk.add(mod)
#for mod in process.filters_().itervalues():
#    process.tsk.add(mod)
process.p = cms.Path(process.selectedAK8Jets*process.countAK8Jets*process.run)
process.p.associate(process.patAlgosToolsTask)
#process.runlast = cms.EndPath(process.selectedAK8Jets+process.run)
#process.runlast = cms.EndPath(process.run)
#process.schedule = cms.Schedule(process.outpath)#, process.runlast)
#patAlgosToolsTask = getPatAlgosToolsTask(process)
#process.schedule.associate(patAlgosToolsTask)

#process.p.associate(process.tsk)
