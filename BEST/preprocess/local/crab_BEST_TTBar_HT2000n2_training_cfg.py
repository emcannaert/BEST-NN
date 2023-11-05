from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BEST_TTBar_HT2000n2_training_0000'
config.General.workArea = 'BESTCrabPreprocess'
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_producerBR.py'
config.Data.inputDataset = '/ZPrimeToTT_M2000_W600_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5
config.Data.outputDatasetTag = 'BEST_TTBar_HT2000'
config.Site.storageSite = 'T3_US_FNALLPC'

