#!/bin/bash

# usage: find_eos_files.sh <main BEST eos folder to search> <year>

# need to split the files into 8 total neural networks:

#2015 low mass
#2015 high mass

#2016 low mass
#2016 high mass

#2017 low mass
#2017 high mass

#2018 low mass
#2018 high mass

###### USAGE: give the name of the eos folder you want to draw files from

EOSBASE="/store/user/ecannaer/"

if [ -z "$1" ];
then
	echo "Invalid crab submission folder. Please provide the most recent crab submission folder on eos (Ex. BESTInputTrees_2024219_161315.....)."
else
	echo "Looking for EOS files for directory $EOSBASE$1"

	xrdfs root://cmseos.fnal.gov ls -R /store/user/ecannaer/$1 | grep 2015 | grep root > all_BEST_files_2015.txt
	echo "Created 2015 files."
	xrdfs root://cmseos.fnal.gov ls -R /store/user/ecannaer/$1 | grep 2016 | grep root > all_BEST_files_2016.txt
	echo "Created 2016 files."
	xrdfs root://cmseos.fnal.gov ls -R /store/user/ecannaer/$1 | grep 2017 | grep root > all_BEST_files_2017.txt
	echo "Created 2017 files."
	xrdfs root://cmseos.fnal.gov ls -R /store/user/ecannaer/$1 | grep 2018 | grep root > all_BEST_files_2018.txt
	echo "Created 2018 files."

	## create 2015 samples

	## TTbar
	grep ToTTJets all_BEST_files_2015.txt  >  SuuToChiChi_Top_2015.txt
	grep TTJets_TuneCP5_13TeV all_BEST_files_2015.txt  >> SuuToChiChi_Top_2015.txt
	grep TTTo all_BEST_files_2015.txt   >> SuuToChiChi_Top_2015.txt
	grep TTbar all_BEST_files_2015.txt  >> SuuToChiChi_Top_2015.txt
	grep ZPrimeTo all_BEST_files_2015.txt >> SuuToChiChi_Top_2015.txt
	grep RSGluonToTT all_BEST_files_2015.txt >> SuuToChiChi_Top_2015.txt

	echo "Filled TTbar files 2015."

	## QCD
	grep QCD all_BEST_files_2015.txt  > SuuToChiChi_QCD_2015.txt
	echo "Filled QCD files 2015."

	##HT
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT | grep 4000  >  SuuToChiChi_HT_low_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT | grep 5000  >> SuuToChiChi_HT_low_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT | grep 6000  >> SuuToChiChi_HT_low_mass_2015.txt	
	echo "Filled low mass HT files 2015."

	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT | grep 6000  >  SuuToChiChi_HT_high_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT | grep 7000  >> SuuToChiChi_HT_high_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT | grep 8000  >> SuuToChiChi_HT_high_mass_2015.txt	
	echo "Filled high mass HT files 2015."

	## WB
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB | grep 4000  >  SuuToChiChi_WB_low_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB | grep 5000  >> SuuToChiChi_WB_low_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB | grep 6000  >> SuuToChiChi_HT_low_mass_2015.txt	
	echo "Filled low mass WB files 2015."

	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB | grep 6000  >  SuuToChiChi_WB_high_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB | grep 7000  >> SuuToChiChi_WB_high_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB | grep 8000  >> SuuToChiChi_WB_high_mass_2015.txt	
	echo "Filled high mass WB files 2015."

	## ZT
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT | grep 4000  >  SuuToChiChi_ZT_low_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT | grep 5000  >> SuuToChiChi_ZT_low_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT | grep 6000  >> SuuToChiChi_ZT_low_mass_2015.txt	
	echo "Filled low mass ZT files 2015."

	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT | grep 6000  >  SuuToChiChi_ZT_high_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT | grep 7000  >> SuuToChiChi_ZT_high_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT | grep 8000  >> SuuToChiChi_ZT_high_mass_2015.txt	
	echo "Filled high mass ZT files 2015."

	## create 2016 samples

	## TTbar
	grep ToTTJets all_BEST_files_2016.txt  >  SuuToChiChi_Top_2016.txt
	grep TTJets_TuneCP5_13TeV all_BEST_files_2016.txt  >> SuuToChiChi_Top_2016.txt
	grep TTTo all_BEST_files_2016.txt   >> SuuToChiChi_Top_2016.txt
	grep TTbar all_BEST_files_2016.txt  >> SuuToChiChi_Top_2016.txt
	grep ZPrimeTo all_BEST_files_2016.txt >> SuuToChiChi_Top_2016.txt
	grep RSGluonToTT all_BEST_files_2016.txt >> SuuToChiChi_Top_2016.txt

	echo "Filled TTbar files 2016."

	## QCD
	grep QCD all_BEST_files_2016.txt  > SuuToChiChi_QCD_2016.txt
	echo "Filled QCD files 2016."

	##HT
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT | grep 4000  >  SuuToChiChi_HT_low_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT | grep 5000  >> SuuToChiChi_HT_low_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT | grep 6000  >> SuuToChiChi_HT_low_mass_2016.txt	
	echo "Filled low mass HT files 2016."

	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT | grep 6000  >  SuuToChiChi_HT_high_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT | grep 7000  >> SuuToChiChi_HT_high_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT | grep 8000  >> SuuToChiChi_HT_high_mass_2016.txt	
	echo "Filled high mass HT files 2016."

	## WB
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB | grep 4000  >  SuuToChiChi_WB_low_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB | grep 5000  >> SuuToChiChi_WB_low_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB | grep 6000  >> SuuToChiChi_HT_low_mass_2016.txt	
	echo "Filled low mass WB files 2016."

	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB | grep 6000  >  SuuToChiChi_WB_high_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB | grep 7000  >> SuuToChiChi_WB_high_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB | grep 8000  >> SuuToChiChi_WB_high_mass_2016.txt	
	echo "Filled high mass WB files 2016."

	## ZT
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT | grep 4000  >  SuuToChiChi_ZT_low_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT | grep 5000  >> SuuToChiChi_ZT_low_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT | grep 6000  >> SuuToChiChi_ZT_low_mass_2016.txt	
	echo "Filled low mass ZT files 2016."

	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT | grep 6000  >  SuuToChiChi_ZT_high_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT | grep 7000  >> SuuToChiChi_ZT_high_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT | grep 8000  >> SuuToChiChi_ZT_high_mass_2016.txt	
	echo "Filled high mass ZT files 2016."


	## create 2017 samples

	## TTbar
	grep ToTTJets all_BEST_files_2017.txt  >  SuuToChiChi_Top_2017.txt
	grep TTJets_TuneCP5_13TeV all_BEST_files_2017.txt  >> SuuToChiChi_Top_2017.txt
	grep TTTo all_BEST_files_2017.txt   >> SuuToChiChi_Top_2017.txt
	grep TTbar all_BEST_files_2017.txt  >> SuuToChiChi_Top_2017.txt
	grep ZPrimeTo all_BEST_files_2017.txt >> SuuToChiChi_Top_2017.txt
	grep RSGluonToTT all_BEST_files_2017.txt >> SuuToChiChi_Top_2017.txt

	echo "Filled TTbar files 2017."

	## QCD
	grep QCD all_BEST_files_2017.txt  > SuuToChiChi_QCD_2017.txt
	echo "Filled QCD files 2017."

	##HT
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT | grep 4000  >  SuuToChiChi_HT_low_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT | grep 5000  >> SuuToChiChi_HT_low_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT | grep 6000  >> SuuToChiChi_HT_low_mass_2017.txt	
	echo "Filled low mass HT files 2017."

	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT | grep 6000  >  SuuToChiChi_HT_high_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT | grep 7000  >> SuuToChiChi_HT_high_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT | grep 8000  >> SuuToChiChi_HT_high_mass_2017.txt	
	echo "Filled high mass HT files 2017."

	## WB
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB | grep 4000  >  SuuToChiChi_WB_low_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB | grep 5000  >> SuuToChiChi_WB_low_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB | grep 6000  >> SuuToChiChi_HT_low_mass_2017.txt	
	echo "Filled low mass WB files 2017."

	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB | grep 6000  >  SuuToChiChi_WB_high_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB | grep 7000  >> SuuToChiChi_WB_high_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB | grep 8000  >> SuuToChiChi_WB_high_mass_2017.txt	
	echo "Filled high mass WB files 2017."

	## ZT
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT | grep 4000  >  SuuToChiChi_ZT_low_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT | grep 5000  >> SuuToChiChi_ZT_low_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT | grep 6000  >> SuuToChiChi_ZT_low_mass_2017.txt	
	echo "Filled low mass ZT files 2017."

	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT | grep 6000  >  SuuToChiChi_ZT_high_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT | grep 7000  >> SuuToChiChi_ZT_high_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT | grep 8000  >> SuuToChiChi_ZT_high_mass_2017.txt	
	echo "Filled high mass ZT files 2017."


	## create 2018 samples

	## TTbar
	grep ToTTJets all_BEST_files_2018.txt  >  SuuToChiChi_Top_2018.txt
	grep TTJets_TuneCP5_13TeV all_BEST_files_2018.txt  >> SuuToChiChi_Top_2018.txt
	grep TTTo all_BEST_files_2018.txt   >> SuuToChiChi_Top_2018.txt
	grep TTbar all_BEST_files_2018.txt  >> SuuToChiChi_Top_2018.txt
	grep ZPrimeTo all_BEST_files_2018.txt >> SuuToChiChi_Top_2018.txt
	grep RSGluonToTT all_BEST_files_2018.txt >> uuToChiChi_Top_2018.txt

	echo "Filled TTbar files 2018."

	## QCD
	grep QCD all_BEST_files_2018.txt  > SuuToChiChi_QCD_2018.txt
	echo "Filled QCD files 2018."

	##HT
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT | grep 4000  >  SuuToChiChi_HT_low_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT | grep 5000  >> SuuToChiChi_HT_low_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT | grep 6000  >> SuuToChiChi_HT_low_mass_2018.txt	
	echo "Filled low mass HT files 2018."

	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT | grep 6000  >  SuuToChiChi_HT_high_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT | grep 7000  >> SuuToChiChi_HT_high_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT | grep 8000  >> SuuToChiChi_HT_high_mass_2018.txt	
	echo "Filled high mass HT files 2018."

	## WB
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB | grep 4000  >  SuuToChiChi_WB_low_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB | grep 5000  >> SuuToChiChi_WB_low_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB | grep 6000  >> SuuToChiChi_HT_low_mass_2018.txt	
	echo "Filled low mass WB files 2018."

	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB | grep 6000  >  SuuToChiChi_WB_high_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB | grep 7000  >> SuuToChiChi_WB_high_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB | grep 8000  >> SuuToChiChi_WB_high_mass_2018.txt	
	echo "Filled high mass WB files 2018."

	## ZT
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT | grep 4000  >  SuuToChiChi_ZT_low_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT | grep 5000  >> SuuToChiChi_ZT_low_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT | grep 6000  >> SuuToChiChi_ZT_low_mass_2018.txt	
	echo "Filled low mass ZT files 2018."

	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT | grep 6000  >  SuuToChiChi_ZT_high_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT | grep 7000  >> SuuToChiChi_ZT_high_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT | grep 8000  >> SuuToChiChi_ZT_high_mass_2018.txt	
	echo "Filled high mass ZT files 2018."


	




	####################################################################################################################################
	####################################################################################################################################
	####################################################################################################################################
	

	##  make training datasets that include all three decays (WB,HT,ZT) and all Suu masses

	## 2015
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT  >   SuuToChiChi_allDecays_all_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB  >>  SuuToChiChi_allDecays_all_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT  >>  SuuToChiChi_allDecays_all_mass_2015.txt	

	## 2016
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT  >   SuuToChiChi_allDecays_all_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB  >>  SuuToChiChi_allDecays_all_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT  >>  SuuToChiChi_allDecays_all_mass_2016.txt	

	## 2017
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT  >   SuuToChiChi_allDecays_all_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB  >>  SuuToChiChi_allDecays_all_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT  >>  SuuToChiChi_allDecays_all_mass_2017.txt	

	## 2018
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT  >   SuuToChiChi_allDecays_all_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB  >>  SuuToChiChi_allDecays_all_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT  >>  SuuToChiChi_allDecays_all_mass_2018.txt	



	##  make training datasets that combine all Suu masses

	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT  >  SuuToChiChi_HT_all_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT  >  SuuToChiChi_ZT_all_mass_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB  >  SuuToChiChi_WB_all_mass_2015.txt	

	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT  >  SuuToChiChi_HT_all_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT  >  SuuToChiChi_ZT_all_mass_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB  >  SuuToChiChi_WB_all_mass_2016.txt	

	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT  >  SuuToChiChi_HT_all_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT  >  SuuToChiChi_ZT_all_mass_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB  >  SuuToChiChi_WB_all_mass_2017.txt	

	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT  >  SuuToChiChi_HT_all_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT  >  SuuToChiChi_ZT_all_mass_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB  >  SuuToChiChi_WB_all_mass_2018.txt	


	## make training datasets for all Suu masses, but just the 1p5 TeV mass point for different decay modes

	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_HT | grep 1500 >  SuuToChiChi_HT_all_mass_Suu1p5_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_WB | grep 1500 >  SuuToChiChi_WB_all_mass_Suu1p5_2015.txt	
	grep SuuToChiChi all_BEST_files_2015.txt | grep BESTInputTrees_ZT | grep 1500 >  SuuToChiChi_ZT_all_mass_Suu1p5_2015.txt	

	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_HT | grep 1500 >  SuuToChiChi_HT_all_mass_Suu1p5_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_WB | grep 1500 >  SuuToChiChi_WB_all_mass_Suu1p5_2016.txt	
	grep SuuToChiChi all_BEST_files_2016.txt | grep BESTInputTrees_ZT | grep 1500 >  SuuToChiChi_ZT_all_mass_Suu1p5_2016.txt	

	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_HT | grep 1500 >  SuuToChiChi_HT_all_mass_Suu1p5_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_WB | grep 1500 >  SuuToChiChi_WB_all_mass_Suu1p5_2017.txt	
	grep SuuToChiChi all_BEST_files_2017.txt | grep BESTInputTrees_ZT | grep 1500 >  SuuToChiChi_ZT_all_mass_Suu1p5_2017.txt	

	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_HT | grep 1500 >  SuuToChiChi_HT_all_mass_Suu1p5_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_WB | grep 1500 >  SuuToChiChi_WB_all_mass_Suu1p5_2018.txt	
	grep SuuToChiChi all_BEST_files_2018.txt | grep BESTInputTrees_ZT | grep 1500 >  SuuToChiChi_ZT_all_mass_Suu1p5_2018.txt	


	#rm all_BEST_files_2015.txt
	#rm all_BEST_files_2016.txt
	#rm all_BEST_files_2017.txt
	#rm all_BEST_files_2018.txt

	echo "Finished."
    #rm all_files.txt
fi






