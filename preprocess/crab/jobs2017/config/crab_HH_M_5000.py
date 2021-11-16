from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")

config.General.requestName = "GravitonHH_5000GeV_trees"
config.General.workArea = "CrabBEST"
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = "Analysis"

config.JobType.psetName = "config/run_HH.py"
#config.JobType.inputFiles = ["TMVARegression_MLP.weights.xml"]
config.JobType.outputFiles = ["BESTInputs.root"]
#config.JobType.allowUndistributedCMSSW = True

config.section_("Data")
config.Data.inputDataset = "/GluGluToBulkGravitonToHHTo4B_M-5000_narrow_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17MiniAOD-106X_mc2017_realistic_v6-v2/MINIAODSIM"
config.Data.splitting = "Automatic"
#config.Data.splitting = "FileBased"
#config.Data.unitsPerJob = 1
config.Data.ignoreLocality = True
config.Data.publication = False
# This string is used to construct the output dataset name

config.section_("Site")
config.Site.storageSite = "T3_US_FNALLPC"
config.Site.whitelist = ["T2_US_*"]
