from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")

config.General.requestName = "ZprimeTT_3500GeV_trees"
config.General.workArea = "CrabBEST"
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = "Analysis"

config.JobType.psetName = "config/run_tt.py"
#config.JobType.inputFiles = ["TMVARegression_MLP.weights.xml"]
config.JobType.outputFiles = ["BESTInputs.root"]
#config.JobType.allowUndistributedCMSSW = True

config.section_("Data")
config.Data.inputDataset = "/ZprimeToTT_M3500_W35_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/MINIAODSIM"
config.Data.splitting = "Automatic"
#config.Data.splitting = "FileBased"
#config.Data.unitsPerJob = 1
config.Data.ignoreLocality = True
config.Data.publication = False
# This string is used to construct the output dataset name

config.section_("Site")
config.Site.storageSite = "T3_US_FNALLPC"
config.Site.whitelist = ["T2_US_*"]
