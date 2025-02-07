from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BESTInputTrees_2015_Top_47_AltDatasets_000'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '../../local/allCfgs//BESTProducer_Top_2015_cfg.py'
config.Data.inputDataset = '/ZPrimeToTT_M5000_W500_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL16MiniAODAPVv2-106X_mcRun2_asymptotic_preVFP_v11-v2/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 10
config.Data.outputDatasetTag = 'BESTInputTrees_Top_2015'
config.Data.outLFNDirBase = '/store/user/tjian/BESTInputTrees_202526_181410'
config.Site.storageSite = 'T3_US_FNALLPC'
