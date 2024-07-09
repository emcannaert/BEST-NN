# BEST for SuuToChiChi : Boosted Event Shape Tagger for the diquark to VLQ to jets analysis 

This is a modified version of the Boosted Event Shape Tagger that trains a neural network to identify vector-like quarks decaying to jets in their center of mass frames. Each VLQ consistes of a collection of jets called superjets. Superjets from signal (comprising all three VLQ decays - Wb, Zt, ht), QCD, and TTbar are used to define the corresponding training categories. 


## Dependencies 

This repository requires CMSSW and python tools for machine learning.

## Installation

This program is written for use with ``CMSSW_10_6_27``. Start installation by installing CMSSW.

```bash
cmsrel CMSSW_10_6_27
cd CMSSW_10_6_27/src/
scram b -j8
```
Then, make a fork, clone the repository, and compile the programs as modules for CMSSW.

```bash
cd CMSSW_10_6_27/src/
git clone https://github.com/emcannaert/BEST-NN.git
scram b -j8
```


## pre-process (creating input TTrees)

The process starts by creating TTrees for each training category that contain all input features (about 130 per event) that will be used for training. TTrees are created by the plugins/BESTProducer.cc analyzer, but the analyzer cfg and crab cfg files need to be created to in order to run and submit with different background and signal samples. These can be created by running the template scripts - 

```bash
cd local
python createCfgTemplateBEST.py
cd ../crab/templates
python createCrabCfgTemplateBEST.py
```

The analyzer cfgs and crab cfgs will be located in the $CMSSW_BASE/src//BEST/preprocess/local/allCfgs/ and $CMSSW_BASE/src/BEST/preprocess/crab/allAltCrabCfgs/ folders respectively. 

The crab cfg files can be submitted, resubmitted, and checked using shell scripts in the allAltCrabCfgs/ folder - 
```bash
cd $CMSSW_BASE/src/BEST/preprocess/crab/allAltCrabCfgs/
source submitCrab_All.sh
source resubmitCrab_All.sh
source checkCrab_All.sh
```

## format conversion (turning TTrees into usable h5 files)

NN training with tensorflow requires inputs in the form of h5 files. The TTrees created in the pre-process step can be converted to this format using scripts in the $CMSSW_BASE/src/BEST/formatConverter folder. The eos paths to the TTrees first need to be collected into to text files in the eosSamples folder, which can be done through the create_sample_lists.sh script - 


```bash
source create_sample_lists.sh <path to eos folder where training TTrees are stored>   # example: /store/user/ecannaer/BESTInputTrees_202445_162215
```

The training events can then be converted, split (into training and test), and flattened - 

```bash
python sampleConverter.py
python sampleSplitter.py
python sampleFlattener.py
```

These output subsequent h5 files locally in the h5samples folder, so be careful about running out of disk space.

## format conversion (turning TTrees into usable h5 files)

The NN must then be trained. 

```bash
python nnBEST.py -r  ## -r overwrites older models
```

This will take a while (a couple hours likely), and the model and plots will be stored in the models/ and plots/ folders respectively. The model file will need to be moved to your main analysis workspace in order to be imported by any analyzers using this NN.

