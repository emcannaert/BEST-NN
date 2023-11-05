from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BEST_Wb_training_M1400_0000'
config.General.workArea = 'BESTCrabPreprocess'
#config.General.transferOutputs = True
#config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_producerWb.py'
config.Data.inputDataset = '/TprimeTprime_M-1400_TuneCP5_PSweights_13TeV-madgraph-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5
#config.Data.lumiMask = 'Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'
config.Data.outputDatasetTag = 'BEST_Wb_training_M1400'
config.Site.storageSite = 'T3_US_FNALLPC'
