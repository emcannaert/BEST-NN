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
process.GlobalTag = GlobalTag(process.GlobalTag, GT)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))


process.source = cms.Source("PoolSource",
        # Replace root file below with the source file you want to use (overwritten by crab config files that call this run file)
        fileNames = cms.untracked.vstring(
        # "/store/mc/RunIISummer20UL16MiniAODv2/GluGluToBulkGravitonToHHTo4B_M-500_narrow_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v2/270000/4464EFA7-46EC-EC4E-88B4-912456E503BD.root"
        # "/store/mc/RunIISummer20UL16MiniAODAPV/GluGluToBulkGravitonToHHTo4B_M-1000_narrow_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v8-v2/120000/865B900D-E5A9-2947-9C52-45F21DD49855.root"
        "/store/mc/RunIISummer20UL17MiniAOD/GluGluToBulkGravitonToHHTo4B_M-4000_narrow_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mc2017_realistic_v6-v2/70000/CC9620A0-A1EE-3C48-B8DF-285766228071.root"
        # "/store/mc/RunIISummer20UL16MiniAODv2/GluGluToBulkGravitonToHHTo4B_M-8000_narrow_TuneCP5_13TeV-madgraph-pythia8/MINIAODSIM/106X_mcRun2_asymptotic_v17-v2/260000/53BD7641-0BEC-FE4F-9961-85BF441D6D86.root"
                                         )
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#=========================================================================================
# Add Deep AK8 variables -----------------------------------------------------------------
#=========================================================================================
updateJetCollection(
   process,
   jetSource = cms.InputTag('slimmedJetsAK8'),
   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
   svSource = cms.InputTag('slimmedSecondaryVertices'),
   rParam = 0.8,
   jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
   btagDiscriminators = ['pfCombinedInclusiveSecondaryVertexV2BJetTags',
                         'pfDeepBoostedJetTags:probTbcq', 'pfDeepBoostedJetTags:probTbqq',
                         'pfDeepBoostedJetTags:probWcq', 'pfDeepBoostedJetTags:probWqq',
                         'pfDeepBoostedJetTags:probZbb', 'pfDeepBoostedJetTags:probZcc', 'pfDeepBoostedJetTags:probZqq',
                         'pfDeepBoostedJetTags:probHbb', 'pfDeepBoostedJetTags:probHcc', 'pfDeepBoostedJetTags:probHqqqq',
                         'pfDeepBoostedJetTags:probQCDbb', 'pfDeepBoostedJetTags:probQCDcc',
                         'pfDeepBoostedJetTags:probQCDb', 'pfDeepBoostedJetTags:probQCDc',
                         'pfDeepBoostedJetTags:probQCDothers',
                         'pfDeepBoostedDiscriminatorsJetTags:TvsQCD', 'pfDeepBoostedDiscriminatorsJetTags:WvsQCD',
                         'pfDeepBoostedDiscriminatorsJetTags:ZvsQCD', 'pfDeepBoostedDiscriminatorsJetTags:ZbbvsQCD',
                         'pfDeepBoostedDiscriminatorsJetTags:HbbvsQCD', 'pfDeepBoostedDiscriminatorsJetTags:H4qvsQCD',
                         'pfMassDecorrelatedDeepBoostedJetTags:probTbcq', 'pfMassDecorrelatedDeepBoostedJetTags:probTbqq',
                         'pfMassDecorrelatedDeepBoostedJetTags:probWcq', 'pfMassDecorrelatedDeepBoostedJetTags:probWqq',
                         'pfMassDecorrelatedDeepBoostedJetTags:probZbb', 'pfMassDecorrelatedDeepBoostedJetTags:probZcc', 'pfMassDecorrelatedDeepBoostedJetTags:probZqq',
                         'pfMassDecorrelatedDeepBoostedJetTags:probHbb', 'pfMassDecorrelatedDeepBoostedJetTags:probHcc', 'pfMassDecorrelatedDeepBoostedJetTags:probHqqqq',
                         'pfMassDecorrelatedDeepBoostedJetTags:probQCDbb', 'pfMassDecorrelatedDeepBoostedJetTags:probQCDcc',
                         'pfMassDecorrelatedDeepBoostedJetTags:probQCDb', 'pfMassDecorrelatedDeepBoostedJetTags:probQCDc',
                         'pfMassDecorrelatedDeepBoostedJetTags:probQCDothers',
                         'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD', 'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD',
                         'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD', 'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHccvsQCD',
                         'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight', 'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ccvsLight'],
    postfix = 'WithDeepTags',
    #postfix = 'AK8',
    printWarning = False # Making this false removes the "b tagging need to be run on uncorrected jets" warning, which would print for every job.
)


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
                             inputJetColl = cms.string('slimmedJetsAK8'),
                             jetColl = cms.string('PUPPI'),                     
							 jetType = cms.string("H"),
                             storeDaughters = cms.bool(True),
)
process.TFileService = cms.Service("TFileService", fileName = cms.string("HH_4000_BESTInputs.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("HH_ana_out.root"),
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
process.p = cms.Path(process.selectedAK8Jets*process.countAK8Jets*process.run)

