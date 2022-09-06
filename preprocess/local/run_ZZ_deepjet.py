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
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTag = GlobalTag(process.GlobalTag, GT)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000))


process.source = cms.Source("PoolSource",
        # Replace root file below with the source file you want to use (overwritten by crab config files that call this run file)
        fileNames = cms.untracked.vstring(
		# "/store/mc/RunIISummer20UL16MiniAODv2/BulkGravToZZToZhadZhad_narrow_M-500_TuneCP5_13TeV-madgraph-pythia/MINIAODSIM/106X_mcRun2_asymptotic_v17-v2/260000/FDB63BDA-822F-3B4C-BDA9-82352D326B69.root"
        # "/store/mc/RunIISummer20UL16MiniAODAPV/BulkGravToZZToZhadZhad_narrow_M-1000_TuneCP5_13TeV-madgraph-pythia/MINIAODSIM/106X_mcRun2_asymptotic_preVFP_v8-v2/140000/D4297079-2764-484E-8154-5A8DDCBB9111.root"
        "file:/afs/cern.ch/work/b/bregnery/public/BESTstudies/CMSSW_10_6_29/src/BEST/preprocess/local/FF6DA8D7-7B1A-D341-A226-19C12EF0621B.root"
        #"/store/mc/RunIISummer20UL17MiniAODv2/BulkGravToZZToZhadZhad_narrow_M-4000_TuneCP5_13TeV-madgraph-pythia/MINIAODSIM/106X_mc2017_realistic_v9-v2/260000/FF6DA8D7-7B1A-D341-A226-19C12EF0621B.root"
        # "/store/mc/RunIISummer20UL16MiniAODv2/BulkGravToZZToZhadZhad_narrow_M-8000_TuneCP5_13TeV-madgraph-pythia/MINIAODSIM/106X_mcRun2_asymptotic_v17-v2/230000/A0ADDB1D-04E2-5246-B4D8-6368556763CB.root"
                                         )
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000




#=========================================================================================
# Add Deep AK8 variables -----------------------------------------------------------------
#=========================================================================================
updateJetCollection(
   process,
   jetSource = cms.InputTag('selectedUpdatedPatJetsNewDFTraining'),
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
# adding b-tagging for subjets -----------------------------------------------------------------            
#=========================================================================================   

useData= False         

bTagInfos = [
    'pfImpactParameterTagInfos', 'pfSecondaryVertexTagInfos', 'pfInclusiveSecondaryVertexFinderTagInfos', 'softPFMuonsTagInfos', 'softPFElectronsTagInfos'
]

# b-tags we want to have for subjet b-tagging
ak4btagDiscriminators = [
    'pfDeepFlavourJetTags:probb',
    'pfDeepFlavourJetTags:probbb',
    'pfDeepFlavourJetTags:problepb',
    'pfDeepFlavourJetTags:probc',
    'pfDeepFlavourJetTags:probuds',
    'pfDeepFlavourJetTags:probg'
]

ak8btagDiscriminators = [
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ccvsLight',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHccvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:H4qvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:HbbvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZbbvsQCD',
    'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags:TvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags:WvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags:H4qvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags:HbbvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags:ZbbvsQCD',
    'pfDeepBoostedDiscriminatorsJetTags:ZvsQCD',
    'pfMassDecorrelatedDeepBoostedJetTags:probHbb',
    'pfMassDecorrelatedDeepBoostedJetTags:probQCDc',
    'pfMassDecorrelatedDeepBoostedJetTags:probQCDbb',
    'pfMassDecorrelatedDeepBoostedJetTags:probTbqq',
    'pfMassDecorrelatedDeepBoostedJetTags:probTbcq',
    'pfMassDecorrelatedDeepBoostedJetTags:probTbq',
    'pfMassDecorrelatedDeepBoostedJetTags:probQCDothers',
    'pfMassDecorrelatedDeepBoostedJetTags:probQCDb',
    'pfMassDecorrelatedDeepBoostedJetTags:probTbc',
    'pfMassDecorrelatedDeepBoostedJetTags:probWqq',
    'pfMassDecorrelatedDeepBoostedJetTags:probQCDcc',
    'pfMassDecorrelatedDeepBoostedJetTags:probHcc',
    'pfMassDecorrelatedDeepBoostedJetTags:probWcq',
    'pfMassDecorrelatedDeepBoostedJetTags:probZcc',
    'pfMassDecorrelatedDeepBoostedJetTags:probZqq',
    'pfMassDecorrelatedDeepBoostedJetTags:probHqqqq',
    'pfMassDecorrelatedDeepBoostedJetTags:probZbb',
    'pfDeepDoubleBvLJetTags:probHbb',
    'pfDeepDoubleBvLJetTags:probQCD',
    'pfDeepDoubleCvBJetTags:probHbb',
    'pfDeepDoubleCvBJetTags:probHcc',
    'pfDeepDoubleCvLJetTags:probHcc',
    'pfDeepDoubleCvLJetTags:probQCD',
    'pfMassIndependentDeepDoubleBvLJetTags:probHbb',
    'pfMassIndependentDeepDoubleBvLJetTags:probQCD',
    'pfMassIndependentDeepDoubleCvBJetTags:probHbb',
    'pfMassIndependentDeepDoubleCvBJetTags:probHcc',
    'pfMassIndependentDeepDoubleCvLJetTags:probHcc',
    'pfMassIndependentDeepDoubleCvLJetTags:probQCD',
    'pfDeepBoostedJetTags:probHbb',
    'pfDeepBoostedJetTags:probQCDc',
    'pfDeepBoostedJetTags:probQCDbb',
    'pfDeepBoostedJetTags:probTbqq',
    'pfDeepBoostedJetTags:probTbcq',
    'pfDeepBoostedJetTags:probTbq',
    'pfDeepBoostedJetTags:probQCDothers',
    'pfDeepBoostedJetTags:probQCDb',
    'pfDeepBoostedJetTags:probTbc',
    'pfDeepBoostedJetTags:probWqq',
    'pfDeepBoostedJetTags:probQCDcc',
    'pfDeepBoostedJetTags:probHcc',
    'pfDeepBoostedJetTags:probWcq',
    'pfDeepBoostedJetTags:probZcc',
    'pfDeepBoostedJetTags:probZqq',
    'pfDeepBoostedJetTags:probHqqqq',
    'pfDeepBoostedJetTags:probZbb'
]


common_btag_parameters = dict(
    #trackSource = cms.InputTag('unpackedTracksAndVertices'),
    pfCandidates=cms.InputTag('packedPFCandidates'),
    pvSource=cms.InputTag('offlineSlimmedPrimaryVertices'),
    svSource=cms.InputTag('slimmedSecondaryVertices'),
    muSource=cms.InputTag('slimmedMuons'),
    elSource=cms.InputTag('slimmedElectrons'),
    btagInfos=bTagInfos,
    btagDiscriminators=ak4btagDiscriminators
)

# captitalize string; needed below to construct pat module names.
def cap(s): return s[0].upper() + s[1:]

task = cms.Task()
from RecoJets.Configuration.RecoPFJets_cff import ak8PFJets

process.ak8PuppiJetsFat = ak8PFJets.clone(
    src=cms.InputTag('puppi'),
    doAreaFastjet=True,
    jetPtMin=150.
    
)
task.add(process.ak8PuppiJetsFat)

# Like for CHS, this makes groomed fatjets as reco::BasicJets,
# and the subjets as reco::PFJets
from RecoJets.Configuration.RecoPFJets_cff import ak8PFJetsCHSSoftDrop
process.ak8PuppiJetsSoftDrop = ak8PFJetsCHSSoftDrop.clone(
    src=cms.InputTag('puppi'),
    jetPtMin=150.
)
task.add(process.ak8PuppiJetsSoftDrop)

# Like for CHS, this makes only the groomed fatjets as reco::PFJets

process.ak8PuppiJetsSoftDropforsub = process.ak8PuppiJetsSoftDrop.clone(
    src=cms.InputTag('puppi')
)
delattr(process.ak8PuppiJetsSoftDropforsub, "writeCompound")
delattr(process.ak8PuppiJetsSoftDropforsub, "jetCollInstanceName")

task.add(process.ak8PuppiJetsSoftDropforsub)

from PhysicsTools.PatAlgos.tools.jetTools import addJetCollection
def add_fatjets_subjets(process, fatjets_name, groomed_jets_name, jetcorr_label='AK8PFchs', jetcorr_label_subjets='AK4PFchs', genjets_name=None, verbose=True, btagging=True, top_tagging=False):
    rParam = getattr(process, fatjets_name).rParam.value()
    algo_dict = {"CambridgeAachen": "ca", "AntiKt": "ak"}
    algo = algo_dict.get(
        getattr(process, fatjets_name).jetAlgorithm.value(), None)
    if algo is None:
        raise RuntimeError, "cannot guess jet algo (ca/ak) from jet producer %s", fatjets_name

    if verbose:
        print '***  Adding fatjets_subjets for', fatjets_name

    subjets_name = groomed_jets_name + 'Subjets'  # e.g. CA8CHSPruned + Subjets

    # add genjet producers, if requested:
    groomed_genjets_name = 'INVALID'
    ungroomed_genjets_name = 'INVALID'

    if genjets_name is not None:
        groomed_jetproducer = getattr(process, groomed_jets_name)
        assert groomed_jetproducer.type_() in ('FastjetJetProducer',
                                               'CATopJetProducer'), "do not know how to construct genjet collection for %s" % repr(groomed_jetproducer)
        groomed_genjets_name = genjets_name(groomed_jets_name)
        if verbose:
            print "  Adding groomed genjets:", groomed_genjets_name
        if not hasattr(process, groomed_genjets_name):
            setattr(process,
                    groomed_genjets_name,
                    groomed_jetproducer.clone(
                        src=cms.InputTag('packedGenParticlesForJetsNoNu'),
                        jetType='GenJet'
                        )
                    )
            task.add(getattr(process, groomed_genjets_name))
        # add for ungroomed jets if not done yet (maybe never used in case
        # ungroomed are not added, but that's ok ..)
        ungroomed_jetproducer = getattr(process, fatjets_name)
        assert ungroomed_jetproducer.type_() == 'FastjetJetProducer', "ungroomed_jetproducer is not a FastjetJetProducer"
        ungroomed_genjets_name = genjets_name(fatjets_name)
        if verbose:
            print "  Adding ungroomed genjets:", ungroomed_genjets_name
        if not hasattr(process, ungroomed_genjets_name):
            setattr(process,
                    ungroomed_genjets_name,
                    ungroomed_jetproducer.clone(
                        src=cms.InputTag('packedGenParticlesForJetsNoNu'),
                        jetType='GenJet'
                        )
                    )
            task.add(getattr(process, ungroomed_genjets_name))

    jetcorr_list = ['L1FastJet', 'L2Relative', 'L3Absolute']
    if useData:
        jetcorr_list.append('L2L3Residual')
    if "puppi" in fatjets_name.lower():
        jetcorr_list = jetcorr_list[1:]

    if jetcorr_label:
        jetcorr_arg = (jetcorr_label, cms.vstring(jetcorr_list), 'None')
    else:
        jetcorr_arg = None

    # patify ungroomed jets, if not already done:
    ungroomed_patname = 'patJets' + cap(fatjets_name)
    add_ungroomed = not hasattr(process, ungroomed_patname)
    if add_ungroomed:
        if verbose:
            print "  Adding ungroomed jets:", ungroomed_patname
        addJetCollection(process,
                         labelName=fatjets_name,
                         jetSource=cms.InputTag(fatjets_name),
                         algo=algo,
                         rParam=rParam,
                         jetCorrections=jetcorr_arg,
                         genJetCollection=cms.InputTag(ungroomed_genjets_name),
                         getJetMCFlavour=not useData,
                         **common_btag_parameters
                         )
        getattr(process, ungroomed_patname).addTagInfos = True
        delattr(process, "selectedPatJets"+cap(fatjets_name))

    # patify groomed fat jets, with b-tagging:
    groomed_patname = "patJets" + cap(groomed_jets_name)
    if verbose:
        print "  Adding groomed jets:", groomed_patname
    addJetCollection(process,
                     labelName=groomed_jets_name,
                     jetSource=cms.InputTag(groomed_jets_name),
                     algo=algo,
                     rParam=rParam,
                     jetCorrections=jetcorr_arg,
                     # genJetCollection = cms.InputTag(groomed_genjets_name), #
                     # nice try, but PAT looks for GenJets, whereas jets with
                     # subjets are BasicJets, so PAT cannot be used for this
                     # matching ...
                     genJetCollection=cms.InputTag("slimmedGenJets"),
                     getJetMCFlavour=not useData,
                     **common_btag_parameters
                     )
    getattr(process, groomed_patname).addTagInfos = True
    if top_tagging:
        getattr(process, groomed_patname).tagInfoSources = cms.VInputTag(groomed_jets_name)
    delattr(process, "selectedPatJets"+cap(groomed_jets_name))

    # patify subjets, with subjet b-tagging:
    subjets_patname = "patJets" + cap(subjets_name)
    if verbose:
        print "  Adding groomed jets' subjets:", subjets_patname
    if jetcorr_label_subjets:
        jetcorr_arg = (jetcorr_label_subjets,
                       cms.vstring(jetcorr_list), 'None')
    else:
        jetcorr_arg = None
    addJetCollection(process,
                     labelName=subjets_name,
                     jetSource=cms.InputTag(groomed_jets_name, 'SubJets'),
                     algo=algo,
                     rParam=rParam,
                     jetCorrections=jetcorr_arg,
                     explicitJTA=True,
                     svClustering=True,
                     fatJets=cms.InputTag(fatjets_name),
                     groomedFatJets=cms.InputTag(groomed_jets_name),
                     genJetCollection=cms.InputTag(
                         groomed_genjets_name, 'SubJets'),
                     getJetMCFlavour=not useData,
                     **common_btag_parameters
#                     **common_btag_parameters_subjet
                     )
    # Always add taginfos to subjets, but possible not to store them,
    # configurable with ntuple writer parameter: subjet_taginfos
    # Attention: Only CVS b-tag info is stored for sub-jets
    getattr(process, subjets_patname).addTagInfos = True
    delattr(process, "selectedPatJets"+cap(subjets_name))


    # Add DeepFlavor b-tagging to sub-jets
    labelName = cap(subjets_patname)
    is_puppi = "puppi" in labelName.lower()

    # # This call to updateJetCollection adds one PATJetUpdater to only remove the JECs,
    # # then uses that as the input to another PATJetUpdater, which re-applies the JECs,
    # # adds in all b tag stuff, etc.
    # # The 2nd PATJetsUpdater has the extra "TransientCorrected" bit in its name.
    # # It also produces a final similar "selectedUpdatedPatJets"+labelName+postfix collection
    # # which is a PATJetSelector
    postfix = ''
    updater_src = "updatedPatJets" + labelName + postfix  # 1st PATJetUpdater, that removes JECs, is src to updater_name
    updater_name = "updatedPatJetsTransientCorrected" + labelName + postfix  # 2nd PATJetUpdater
    selector_name = "selectedUpdatedPatJets" + labelName + postfix
    if is_puppi:
        correction_tag = "AK4PFPuppi"
    else:
        correction_tag = "AK4PFchs"

    jetcorr_list = ['L1FastJet', 'L2Relative', 'L3Absolute']
    if is_puppi:
        jetcorr_list = jetcorr_list[1:]
    if useData:
        jetcorr_list.append("L2L3Residual")
    discriminators = ak4btagDiscriminators[:]

    updateJetCollection(
        process,
        labelName=labelName,
        jetSource=cms.InputTag(subjets_patname),
        pvSource=cms.InputTag('offlineSlimmedPrimaryVertices'),
        svSource=cms.InputTag('slimmedSecondaryVertices'),
        jetCorrections=jetcorr_arg,
        btagDiscriminators=discriminators,
        postfix=postfix,
        printWarning=False
    )

    subjets_patname = "updatedPatJetsTransientCorrected" + cap(subjets_patname)

    # # Rekey subjets so constituents point to packedPFCandidates not PUPPI
    # # do before BoostedJetMerger otherwise painful afterwards
    # # Don't rekey groomed fatjet as we don't care about it - the
    # # BoostedJetMerger will replace its daughters with subjets anyway
    # subjets_rekey_name = "rekey"+cap(subjets_patname)
    subjets_rekey_name = subjets_patname
    # setattr(process,
    #         subjets_rekey_name,
    #         cms.EDProducer("RekeyJets",
    #                         jetSrc=cms.InputTag(subjets_patname),
    #                         candidateSrc=cms.InputTag("packedPFCandidates"),
    #                         )
    #         )
    # task.add(getattr(process, subjets_rekey_name))

    # add the merged jet collection which contains the links from groomed
    # fat jets to the subjets:
    groomed_packed_name = groomed_patname + 'Packed'
    if verbose:
        print "  Adding groomed jets + subjets packer:", groomed_packed_name
    setattr(process,
            groomed_packed_name,
            cms.EDProducer("BoostedJetMerger",
                            jetSrc=cms.InputTag(groomed_patname),
                            subjetSrc=cms.InputTag(subjets_rekey_name)
                          )
            )
    task.add(getattr(process, groomed_packed_name))

    # adapt all for b-tagging, and switch off some PAT features not supported
    # in miniAOD:
    module_names = [subjets_name, groomed_jets_name]
    if add_ungroomed:
        module_names += [fatjets_name]
    for name in module_names:
        getattr(process, 'patJetPartonMatch' + cap(name)).matched = 'prunedGenParticles'
        producer = getattr(process, 'patJets' + cap(name))
        producer.addJetCharge = False
        producer.addAssociatedTracks = False
        if not btagging:
            producer.addDiscriminators = False
            producer.addBTagInfo = False
            producer.getJetMCFlavour = False
        producer.addGenJetMatch = genjets_name is not None
        # for fat groomed jets, gen jet match and jet flavor is not working, so
        # switch it off:
        if name == groomed_jets_name:
            producer.addGenJetMatch = False
            producer.getJetMCFlavour = False
        # For data, turn off every gen-related part - can't do this via addJetCollection annoyingly
        if useData:
            modify_patjetproducer_for_data(process, producer)


add_fatjets_subjets(process, 'ak8PuppiJetsFat', 'ak8PuppiJetsSoftDrop',
                    genjets_name=lambda s: s.replace('Puppi', 'Gen'),
                    jetcorr_label="AK8PFPuppi", jetcorr_label_subjets="AK4PFPuppi")


# Packing substructure: fat jets + subjets
#
# Add subjets from groomed fat jet to its corresponding ungroomed fatjet

# # Rekey jets so ungroomed constituents point to packedPFCandidates not PUPPI.
# # Easiet here after all btagging etc calculations done that involve constituents
# # to avoid incorrect calculations
# process.rekeyPatJetsAk8PuppiJetsFat = cms.EDProducer("RekeyJets",
#                                                      jetSrc=cms.InputTag("patJetsAk8PuppiJetsFat"),
#                                                      candidateSrc=cms.InputTag("packedPFCandidates"),
# )
# task.add(process.rekeyPatJetsAk8PuppiJetsFat)

process.rekeyPackedPatJetsAk8PuppiJets = cms.EDProducer("JetSubstructurePacker",
                                                        jetSrc = cms.InputTag("patJetsAk8PuppiJetsFat"),
                                                    distMax = cms.double(0.8),
                                                        algoTags = cms.VInputTag(
                                                            cms.InputTag("patJetsAk8PuppiJetsSoftDropPacked")
                                                        ),
                                                        algoLabels = cms.vstring(
                                                            'SoftDropPuppi'
                                                    ),
                                                        fixDaughters = cms.bool(False)
)
task.add(process.rekeyPackedPatJetsAk8PuppiJets)

###############################################
# Do deep flavours & deep tagging
# This MUST be run *After* JetSubstructurePacker, so that the subjets are already there,
# otherwise the DeepBoostedJetTagInfoProducer will fail
# Also add in PUPPI multiplicities while we're at it.
for name in ['slimmedJets', 'slimmedJetsPuppi', 'rekeyPatJetsAK8PFPUPPI', 'rekeyPackedPatJetsAk8PuppiJets','packedPatJetsAk8CHSJets']:
    labelName = cap(name)
    name_lower = name.lower()
    is_ak8 = "ak8" in name_lower
    is_puppi = "puppi" in name_lower
    is_reclustered = "slimmed" not in name_lower and 'rekey' not in name_lower
    is_topjet = "packed" in name_lower
    # This postfix is VERY IMPORTANT for reclustered puppi, as the puppi weights
    # are already applied. If it doesn't have this postfix then it will apply
    # puppi weights - necessary for slimmedCollections & rekeyed ones
    # (i.e. they have the packedPFCandidates as daughters - require puppi weights to be applied)
    # See https://github.com/cms-sw/cmssw/blob/CMSSW_10_2_10/PhysicsTools/PatAlgos/python/tools/jetTools.py#L653
    # Please check in future releases if this is still the case!
    # It's needed only for pfDeepBoostedJetTagInfos
    postfix = "WithPuppiDaughters" if is_puppi and is_reclustered else "NewDFTraining"  # NewDFTraining is not special, could just be ''

    # This call to updateJetCollection adds one PATJetUpdater to only remove the JECs,
    # then uses that as the input to another PATJetUpdater, which re-applies the JECs,
    # adds in all b tag stuff, etc.
    # The 2nd PATJetsUpdater has the extra "TransientCorrected" bit in its name.
    # It also produces a final similar "selectedUpdatedPatJets"+labelName+postfix collection
    # which is a PATJetSelector
    updater_src = "updatedPatJets" + labelName + postfix  # 1st PATJetUpdater, that removes JECs, is src to updater_name
    updater_name = "updatedPatJetsTransientCorrected" + labelName + postfix  # 2nd PATJetUpdater
    selector_name = "selectedUpdatedPatJets" + labelName + postfix

    if is_ak8 and is_puppi:
        correction_tag = "AK8PFPuppi"
    elif is_ak8 and not is_puppi:
        correction_tag = "AK8PFchs"
    elif not is_ak8 and is_puppi:
        correction_tag = "AK4PFPuppi"
    elif not is_ak8 and not is_puppi:
        correction_tag = "AK4PFchs"

    else:
        raise RuntimeError("No idea which jet correction tag you need here")

    jetcorr_list = ['L1FastJet', 'L2Relative', 'L3Absolute']
    if is_puppi:
        jetcorr_list = jetcorr_list[1:]
    if useData:
        jetcorr_list.append("L2L3Residual")

    discriminators = ak4btagDiscriminators[:]

    if is_ak8 and is_topjet:
        discriminators.extend(ak8btagDiscriminators)

    updateJetCollection(
        process,
        labelName=labelName,
        jetSource=cms.InputTag(name),
        pvSource=cms.InputTag('offlineSlimmedPrimaryVertices'),
        svSource=cms.InputTag('slimmedSecondaryVertices'),
        jetCorrections=(correction_tag, cms.vstring(jetcorr_list), 'None'),  # Can't use None here as we are doing btagging for some reason.
        btagDiscriminators=discriminators,
        postfix=postfix,
        printWarning=False
    )

    if is_puppi:
        # Add puppi multiplicity producers
        # For each, we have to add a PATPuppiJetSpecificProducer,
        # then update the relevant pat::Jet collection using updateJetCollection
        # using userFloats mechanism
        # Crucially, the PATPuppiJetSpecificProducer module name MUST be the same
        # as the final jet collection, with only "patPuppiJetSpecificProducer" prepended
        # so that NtupleWriterJets can find the stored userFloat
        puppi_mult_name = "patPuppiJetSpecificProducer" + updater_name
        setattr(process,
                puppi_mult_name,
                cms.EDProducer("PATPuppiJetSpecificProducer",
                               src = cms.InputTag(updater_src)
                               )
                )
        task.add(getattr(process, puppi_mult_name))


        # We add in the userFloats to the last PATJetUpdater
        # This is because we use the jet collection name to access the userFloats in NtupleWriterJets
        getattr(process, updater_name).userData.userFloats.src = [
            '%s:puppiMultiplicity' % puppi_mult_name,
            '%s:neutralPuppiMultiplicity' % puppi_mult_name,
            '%s:neutralHadronPuppiMultiplicity' % puppi_mult_name,
            '%s:photonPuppiMultiplicity' % puppi_mult_name,
            '%s:HFHadronPuppiMultiplicity' % puppi_mult_name,
            '%s:HFEMPuppiMultiplicity' % puppi_mult_name
            ]


    
#=========================================================================================            
# Add ParticleNet variables -----------------------------------------------------------------            
#=========================================================================================            
#updateJetCollection(
#   process,
#   jetSource = cms.InputTag('slimmedJetsAK8'),
#   pvSource = cms.InputTag('offlineSlimmedPrimaryVertices'),
#   svSource = cms.InputTag('slimmedSecondaryVertices'),
#   rParam = 0.8,
#   jetCorrections = ('AK8PFPuppi', cms.vstring(['L2Relative', 'L3Absolute']), 'None'),
#   btagDiscriminators = ['pfCombinedInclusiveSecondaryVertexV2BJetTags',
#                         'pfParticleNetJetTags:probTbcq', 'pfParticleNetJetTags:probTbqq',
#                         'pfParticleNetJetTags:probWcq', 'pfParticleNetJetTags:probWqq',
#                         'pfParticleNetJetTags:probZbb', 'pfParticleNetJetTags:probZcc', 'pfParticleNetJetTags:probZqq',
#                         'pfParticleNetJetTags:probHbb', 'pfParticleNetJetTags:probHcc', 'pfParticleNetJetTags:probHqqqq',
#                         'pfParticleNetJetTags:probQCDbb', 'pfParticleNetJetTags:probQCDcc',
#                         'pfParticleNetJetTags:probQCDb', 'pfParticleNetJetTags:probQCDc',
#                         'pfParticleNetJetTags:probQCDothers',
   #                      'pfDeepBoostedDiscriminatorsJetTags:TvsQCD', 'pfDeepBoostedDiscriminatorsJetTags:WvsQCD',
   #                      'pfDeepBoostedDiscriminatorsJetTags:ZvsQCD', 'pfDeepBoostedDiscriminatorsJetTags:ZbbvsQCD',
   #                      'pfDeepBoostedDiscriminatorsJetTags:HbbvsQCD', 'pfDeepBoostedDiscriminatorsJetTags:H4qvsQCD',
   #                      'pfMassDecorrelatedParticleNetTags:probTbcq', 'pfMassDecorrelatedParticleNetTags:probTbqq',
   #                      'pfMassDecorrelatedParticleNetTags:probWcq', 'pfMassDecorrelatedParticleNetTags:probWqq',
   #                      'pfMassDecorrelatedParticleNetTags:probZbb', 'pfMassDecorrelatedParticleNetTags:probZcc', 'pfMassDecorrelatedParticleNetTags:probZqq',
   #                      'pfMassDecorrelatedParticleNetTags:probHbb', 'pfMassDecorrelatedParticleNetTags:probHcc', 'pfMassDecorrelatedParticleNetTags:probHqqqq',
   #                      'pfMassDecorrelatedParticleNetTags:probQCDbb', 'pfMassDecorrelatedParticleNetTags:probQCDcc',
   #                      'pfMassDecorrelatedParticleNetTags:probQCDb', 'pfMassDecorrelatedParticleNetTags:probQCDc',
   #                      'pfMassDecorrelatedDeepBoostedJetTags:probQCDothers',
   #                      'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:TvsQCD', 'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:WvsQCD',
   #                      'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHbbvsQCD', 'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ZHccvsQCD',
   #                      'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:bbvsLight', 'pfMassDecorrelatedDeepBoostedDiscriminatorsJetTags:ccvsLight'
#   ],
#    postfix = 'ParticleNet',
    #postfix = 'AK8',                                                                                 
#    printWarning = False # Making this false removes the "b tagging need to be run on uncorrected jets" warning, which would print for every job.                                                          
#)





#=========================================================================================
# Prepare and run producer ---------------------------------------------------------------
#=========================================================================================

# Apply a preselction
jetcollection_name = "updatedPatJetsTransientCorrectedRekeyPackedPatJetsAk8PuppiJetsNewDFTraining"
#jetcollection_name = "patJetsAk8PuppiJetsFat"

process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
                                        src = cms.InputTag("slimmedJetsAK8"),
                                        cut = cms.string('500.0 < pt && pt < 3500.0 && abs(eta) < 2.4'),
                                        filter = cms.bool(True)
)

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
                                    minNumber = cms.uint32(1),
                                    maxNumber = cms.uint32(99999),
                                    src = cms.InputTag("slimmedJetsAK8")
                                    #filter = cms.bool(True)
)


# # Run the producer
process.run = cms.EDProducer('BESTProducer',
#                             inputJetColl = cms.string(jetcollection_name),
                              #inputJetColl = cms.string("updatedPatJetsTransientCorrectedPatJetsAk8PuppiJetsSoftDropSubjets"),
                              inputJetColl = cms.string("updatedPatJetsTransientCorrectedRekeyPackedPatJetsAk8PuppiJetsNewDFTraining"),
#                             inputJetColl = cms.string("slimmedJetsAK8"),
                             jetColl = cms.string('PUPPI'),                     
							 jetType = cms.string("Z"),
                             storeDaughters = cms.bool(True),
)
process.TFileService = cms.Service("TFileService", fileName = cms.string("ZZ_4000_BESTInputs.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ZZ_ana_out.root"),
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_fixedGridRhoAll_*_*',
                                                                      'keep *_run_*_*',
#                                                                      'keep *_*Puppi*_*_*'
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
)
process.outpath = cms.EndPath(process.out)

# Organize the running procedure
process.p = cms.Path(process.selectedAK8Jets*process.countAK8Jets*process.run)
process.p.associate(task)
process.p.associate(process.patAlgosToolsTask)
