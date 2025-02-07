from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BESTInputTrees_2017_ZT_13_AltDatasets_000'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '../../local/allCfgs//BESTProducer_ZT_2017_cfg.py'
config.Data.inputDataset = '/SuuToChiChiToHTZTToJets_MSuu-8000_MChi-1500_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.outputDatasetTag = 'BESTInputTrees_ZT_2017'
config.Data.outLFNDirBase = '/store/user/tjian/BESTInputTrees_202526_181410'
config.Site.storageSite = 'T3_US_FNALLPC'
