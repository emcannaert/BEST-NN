from WMCore.Configuration import Configuration
config = Configuration()

config.section_("General")

# This line will be replaced by createConfig.py: config.General.requestName = "CRABDIRFLAG"
config.General.workArea = "CrabBEST"
config.General.transferLogs = True

config.section_("JobType")
config.JobType.pluginName = "Analysis"

# This line will be replaced by createConfig.py: config.JobType.psetName  = "RUNPARTICLEFLAG"
#config.JobType.inputFiles = ["TMVARegression_MLP.weights.xml"]
config.JobType.outputFiles = ["BESTInputs.root"]
#config.JobType.allowUndistributedCMSSW = True

config.section_("Data")
# This line will be replaced by createConfig.py: config.Data.inputDataset = "DATASETFLAG"
config.Data.splitting = "Automatic"
# This line will be replaced by createConfig.py: config.Data.totalUnits   = "MAXUNITSFLAG"
config.Data.ignoreLocality = True
config.Data.publication = False
# This string is used to construct the output dataset name

config.section_("Site")
config.Site.storageSite = "T3_US_FNALLPC"
config.Site.whitelist = ["T2_US_*"]
