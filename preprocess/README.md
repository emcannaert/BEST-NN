# Preprocessing for BEST

This ED Producer preprocesses CMS Monte Carlo samples. After preprocessing, these datasets 
can be used to train BEST. In the context of this software package, preprocessing means
reducing the size of the input data set by organizing TTrees by jet, then performing preselection
on those jets and matching to gen particles, and finally calculating and storing only the variables
of interest to BEST.

## Overview

The actual producer is located in the ``plugins/BESTProducer.cc``. 
Config file templates are found in ``crab/templates/``, 
which is used by ``crab/buildConfig.py`` to generate the run and config files.
This script is called by ``crab/crabSubmit.sh``, which generates and submits all crab jobs.

# Instructions for Preprocessing

The preprocessing program can be run locally or through CRAB.

## Local Instructions

To run, use the cms environment to run a ``run_*.py`` file. For example: 

```bash
cmsenv
cd local/
cmsRun run_ZZ.py
```

Be sure to update any file locations in the ``run_*.py`` files!!

## CRAB Instructions

First, set up the CRAB environment and obtain a proxy. 
Then, fetch the samples from DAS and submit the CRAB jobs.
By default, the submission scripts submit all jobs:

```bash
cmsenv
cd crab/
source /cvmfs/cms.cern.ch/crab3/crab.sh
voms-proxy-init --voms cms --valid 168:00
./fillSamples.sh
./crabSubmit.sh
``` 

Pass arguments to ``fillSamples.sh`` to select which particles, years, or datatype to check DAS for.
Pass 'all' to an option to select every arguement for that option.
Note that quotes are neccessary to when passing multiple arguements for one option:

```bash
./fillSamples.sh -y "2016_APV 2016 2017" -d data -p all    
```

Run ``./fillSamples.sh -help`` for more information.

To select which particles, years, or datatypes to submit to CRAB, modify the first few lines of ``crabSubmit.sh``.
These variables control which jobs are submitted:

```bash
declare -a myParticles=("HH" "WW" "ZZ" "tt" "bb" "QCD")
declare -a myYears=("2016_APV" "2016" "2017" "2018")
declare -a myDatatypes=("mc" "data")
```

The output files should be ``BESTInputs_X.root`` in the eos location specified in the crab config script. DAS datasets can also be updated inside the ``crab_*.py`` files.

### Useful CRAB Commands

To test, get estimates, and then submit do a crab dryrun

```bash
cd crab/submit2017/
crab submit --dryrun crab_*.py
crab proceed
```

To check the status of a specific job

```
cd crab/submit2017/
crab status CrabBEST/<project_directory>
```

To check all jobs, use the shell script

```
cd crab/
./crabStatus.sh
```

The output can be found at ``logStatus.txt``

To resubmit a specific job

```bash
cd crab/submit2017/
crab resubmit CrabBEST/<project_directory>
```

To resubmit all valid jobs, use the shell script

```
cd crab/
./crabResubmit.sh
```

The output can be found at ``logResubmit.txt``

To kill a specific job of a submission

```
cd crab/submit2017/
crab kill CrabBEST/<project_directory>
```

To kill all jobs, use the shell script

```
cd crab
./crabKill.sh
```

The output can be found at ``logKill.txt``