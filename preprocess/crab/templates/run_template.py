#=========================================================================================
# run_template.py ------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# Author(s): Brendan Regnery, Samantha Abbott, Johan Bonilla, Sydney Ostrom, Reyer Band, -
#            Congqiao Li, Anna Benecke ---------------------------------------------------
#-----------------------------------------------------------------------------------------

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.jetTools import updateJetCollection
from Configuration.AlCa.GlobalTag import GlobalTag

# This line will be replaced by createConfig.py: GT = "GLOBALTAGFLAG"
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

# Option to set max events, used for local cmsRun jobs
# process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
        # Replace root file below with the source file you want to use (overwritten by crab config files that call this run file)
        fileNames = cms.untracked.vstring(
                'myfile.root'
                                         )
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#=========================================================================================
# Add deep flavour b discriminants -------------------------------------------------------
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
    postfix='AK8DF'
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
                                    src = cms.InputTag('selectedAK8Jets')
                                    #filter = cms.bool(True)
)


# Run the producer
# Run the producer
process.run = cms.EDProducer('BESTProducer',
                             inputJetColl = cms.string('selectedAK8Jets'),
                             inputSubJetColl = cms.string('updatedPatJetsTransientCorrectedSoftDropSubjetsPFAK8DF'),
                             jetColl = cms.string('PUPPI'),                     
                             # This line will be replaced by createConfig.py: jetType = cms.string("PARTICLESTRINGFLAG")
                             storeDaughters = cms.bool(True),
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("BESTInputs.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ana_out.root"),
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
process.p.associate(process.patAlgosToolsTask)

