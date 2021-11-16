from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")

config.General.requestName = "GravitonWW_3000GeV_trees"
config.General.workArea = "CrabBEST"
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = "Analysis"

config.JobType.psetName = "config/run_WW.py"
#config.JobType.inputFiles = ["TMVARegression_MLP.weights.xml"]
config.JobType.outputFiles = ["BESTInputs.root"]
#config.JobType.allowUndistributedCMSSW = True

config.section_("Data")
config.Data.inputDataset = "/BulkGravToWWToWhadWhad_narrow_M-3000_TuneCP5_13TeV-madgraph-pythia/RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v2/MINIAODSIM"
config.Data.splitting = "Automatic"
#config.Data.splitting = "FileBased"
#config.Data.unitsPerJob = 1
config.Data.ignoreLocality = True
config.Data.publication = False
# This string is used to construct the output dataset name

config.section_("Site")
config.Site.storageSite = "T3_US_FNALLPC"
config.Site.whitelist = ["T2_US_*"]
