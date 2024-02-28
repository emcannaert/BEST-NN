#! /usr/bin/env python

import sys
import os
from datetime import datetime
import pickle
import numpy as np
### this can be updated to create the different cfgs for each systematic
def makeAltCrabCfg(year, dataset,dateTimeString, jetType, cfg_num):

	file_base = ""
	path_backtrack = ""
	extra_cfg_folderr = ""

	newCfg = open("../allAltCrabCfgs/crab_BEST_Producer_%s_%s_%s_cfg.py"%(year,jetType, cfg_num),"w")

	newCfg.write("from CRABClient.UserUtilities import config\n")
	newCfg.write("config = config()\n")
	newCfg.write("config.General.requestName = 'BESTInputTrees_%s_%s_%s_AltDatasets_000'\n"%(year,jetType, cfg_num))
	newCfg.write("config.General.workArea = 'crab_projects'\n")
	newCfg.write("config.General.transferOutputs = True\n")
	newCfg.write("config.JobType.allowUndistributedCMSSW = True\n")
	newCfg.write("config.JobType.pluginName = 'Analysis'\n")
	
	newCfg.write("config.JobType.psetName = '../../python/allCfgs/BESTProducer_%s_%s_cfg.py'\n"%(jetType, year))

	newCfg.write("config.Data.inputDataset = '%s'\n"%dataset.strip())
	newCfg.write("config.Data.publication = False\n")
	#if "data" in sample:
		#newCfg.write("config.Data.splitting = 'Automatic'\n")
	#else:
	newCfg.write("config.Data.splitting = 'FileBased'\n")
	if jetType == "Top":
		newCfg.write("config.Data.unitsPerJob = 10\n")
	elif jetType == "QCD":
		newCfg.write("config.Data.unitsPerJob = 1\n")
	else:
		newCfg.write("config.Data.unitsPerJob = 1\n")
	if jetType == "QCD":
		newCfg.write("config.JobType.maxMemoryMB = 3000 # might be necessary for some of the QCD jobs\n")

	newCfg.write("config.Data.outputDatasetTag = 'BESTInputTrees_%s_%s'\n"%(jetType,year))
	newCfg.write("config.Data.outLFNDirBase = '/store/user/ecannaer/BESTInputTrees_%s'\n"%dateTimeString)
	newCfg.write("config.Site.storageSite = 'T3_US_FNALLPC'\n")




def main():

	lastCrabSubmission = open("lastCrabSubmission.txt", "a")
	TTbar_datasets = ["TTToHadronic", "ZprimeToTTJets","TTJets", "ZprimeDMToTTbar"]


	

	current_time = datetime.now()
	dateTimeString = "%s%s%s_%s%s%s"%(current_time.year,current_time.month,current_time.day,current_time.hour,current_time.minute,current_time.second )
	lastCrabSubmission.write("/store/user/ecannaer/BESTInputTrees_%s\n"%dateTimeString)
	years   = ["2015","2016","2017","2018"]
	jetTypes = ["WB", "HT", "ZT", "Top", "QCD"]
	num_files_created = 0
	for year in years:
		
		datasets = []
		## get all datset files, compile them into a single dataset
		with open("training_datasets/SuuToChiChi_datasets_%s.txt"%year) as f:
			sig_files = f.readlines()
		with open("training_datasets/NN_TTbar_training_%s.txt"%year) as f:
			Top_files = f.readlines()
		with open("training_datasets/NN_QCD_training_%s.txt"%year) as f:
			QCD_files = f.readlines()
		datasets.extend(sig_files)
		datasets.extend(Top_files)
		datasets.extend(QCD_files)
		cfg_num = {"WB":0, "HT":0, "ZT":0, "QCD":0,"Top":0}
		for jetType in jetTypes:
			for dataset in datasets:
					if jetType != "Top" and jetType in dataset:
						makeAltCrabCfg(year, dataset,dateTimeString, jetType, cfg_num[jetType])  
						cfg_num[jetType]+=1
						num_files_created+=1
					if jetType == "Top": 		# the top datasets are named a number of different things
						for TTbar_dataset in TTbar_datasets:
							if TTbar_dataset in dataset:
								makeAltCrabCfg(year, dataset,dateTimeString, jetType, cfg_num[jetType])  
								cfg_num[jetType]+=1
								num_files_created+=1


	print("Created %i cfg files."%num_files_created)
	return

if __name__ == "__main__":
	main()
