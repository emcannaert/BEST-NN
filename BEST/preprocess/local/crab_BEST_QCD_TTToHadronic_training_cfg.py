from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BEST_TTToHadronic_training_0008'
config.General.workArea = 'BESTCrabPreprocess'
#config.General.transferOutputs = True
#config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_producerBR.py'
#config.Data.inputDataset = '/TTToHadronic_TuneCP5CR1_13TeV-powheg-pythia8/RunIISummer19UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v1/MINIAODSIM'
config.Data.inputDataset = '/TTToHadronic_TuneCP5CR2_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM'
#config.Data.inputDataset = '/TprimeTprime_M-1800_TuneCP5_PSweights_13TeV-madgraph-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5
#config.Data.lumiMask = 'Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'
config.Data.outputDatasetTag = 'BEST_TTToHadronic_training'
config.Site.storageSite = 'T3_US_FNALLPC'
