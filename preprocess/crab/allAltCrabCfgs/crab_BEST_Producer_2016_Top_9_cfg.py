from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BESTInputTrees_2016_Top_9_AltDatasets_000'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '../../local/allCfgs//BESTProducer_Top_2016_cfg.py'
config.Data.inputDataset = '/TTToHadronic_mtop175p5_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v1/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 10
config.Data.outputDatasetTag = 'BESTInputTrees_Top_2016'
config.Data.outLFNDirBase = '/store/user/tjian/BESTInputTrees_202526_181410'
config.Site.storageSite = 'T3_US_FNALLPC'
