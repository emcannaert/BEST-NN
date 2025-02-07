from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BESTInputTrees_2016_ST_5_AltDatasets_000'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '../../local/allCfgs//BESTProducer_ST_2016_cfg.py'
config.Data.inputDataset = '/ST_t-channel_top_4f_InclusiveDecays_mtop1715_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v3/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.outputDatasetTag = 'BESTInputTrees_ST_2016'
config.Data.outLFNDirBase = '/store/user/tjian/BESTInputTrees_202526_181410'
config.Site.storageSite = 'T3_US_FNALLPC'
