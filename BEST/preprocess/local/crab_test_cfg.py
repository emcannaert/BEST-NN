from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'BEST_test_Suu8TeV_chi3TeV_new_Ht'
config.General.workArea = 'BESTCrabPreprocess'
#config.General.transferOutputs = True
#config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'run_producerHt.py'
config.Data.inputDBS = 'phys03'
config.Data.inputDataset = '/SuuToChiChi/ecannaer-Suu8TeV_chi3TeV_ALLDECAYS_TuneCP5_13TeV-pythia8_B2G-RunIISummer20UL18p_MINIAOD-07bb2832fd9cf08ee8da01c42829422a/USER'
#config.Data.inputDataset = '/TprimeTprime_M-1800_TuneCP5_PSweights_13TeV-madgraph-pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
config.Data.publication = False
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 5
#config.Data.lumiMask = 'Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt'
config.Data.outputDatasetTag = 'crab_test_Suu8TeV_Ht'
config.Site.storageSite = 'T3_US_FNALLPC'
