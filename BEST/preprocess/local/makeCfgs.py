#! /usr/bin/env python

import sys

##############################################################################
#MAIN
##############################################################################
def main(argv): 

	datasets = ["/ZPrimeToTT_M1000_W100_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M1200_W120_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M1400_W420_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M1600_W160_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M1600_W480_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M1800_W180_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M1800_W540_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M2000_W200_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M2000_W600_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M2500_W250_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M2500_W750_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M3000_W300_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M3000_W900_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M3500_W1050_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M4000_W1200_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M4000_W400_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M4500_W450_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM",
	"/ZPrimeToTT_M5000_W500_TuneCP2_13TeV-madgraph-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"]


	masses = [1000,1200,1400,1600,1600,1800,1800,2000,2000,2500,2500,3000,3000,3500,4000,4000,4500,5000]
	suffix = [  "" ,"",  ""  ,"",  ""  ,""  ,"2", "" ,"n2" ,"" ,"n2" ,"" ,"n2", ""  ,""  ,"n2", "",  ""]
	for index,mass in enumerate(masses):
		file = open("crab_BEST_TTBar_HT%i%s_training_cfg.py"%(mass,suffix[index]),"w")
		file.write("from CRABClient.UserUtilities import config\n")
		file.write("config = config()\n")
		file.write("config.General.requestName = 'BEST_TTBar_HT%i%s_training_0000'\n"%(mass,suffix[index]))
		file.write("config.General.workArea = 'BESTCrabPreprocess'\n")
		file.write("config.JobType.pluginName = 'Analysis'\n")
		file.write("config.JobType.psetName = 'run_producerBR.py'\n")
		file.write("config.Data.inputDataset = '%s'\n"%datasets[index])
		file.write("config.Data.publication = False\n")
		file.write("config.Data.splitting = 'FileBased'\n")
		file.write("config.Data.unitsPerJob = 5\n")
		file.write("config.Data.outputDatasetTag = 'BEST_TTBar_HT%i'\n"%mass)
		file.write("config.Site.storageSite = 'T3_US_FNALLPC'\n")
		file.write("\n")
if __name__ == "__main__":
	main(sys.argv[1:])


