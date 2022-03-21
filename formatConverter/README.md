# Format Converter

## Overview

The Format Converter creates python h5 files from input root ntuples so that the images and BES variables are in a proper format for training the BEST neural network.

## Conversion Instructions

The conversion takes place using uproot to create useful python data structures. First, make sure that there are directories to store
the h5 files and image plots. Consider making the output h5sample directory in your nobackup space.

```bash
cd ~/nobackup/PATHtoBEST/formatConverter/
mkdir plots
mkdir h5samples
```

Make sure that you have `cmsenv` enabled and have a `vprox`. Next, the lists of eos files must be created using the
`eosSamples/listSamples.sh`. If you have created new `.root` ntuples, be sure to update the search paths in this
shell script.

```bash
cmsenv
cd eosSamples
source listSamples.sh all
```

Use the `sampleConverter.py` script to convert the eos root files to local h5 files:

```bash
python sampleConverter.py -s all -y all -eos <path/to/eosSampleListsDir/> -o <path/to/h5samples/>
```

## Splitting and Flattening

Next, split the sample into train, validation, and test sets with 'sampleSplitter.py'.
Then flatten (shape-match by pT) each of those output files. See each file for full list of arguments.

```bash
python sampleSplitter.py -s all -hd </path/to/h5samples/> -o </path/to/outputH5samples> -bs <batchSize=(example)600000>
python flattener.py -s all -st train,validation,test -hd </path/to/Inputh5samples/> -o </path/to/Outputh5samples/> -b <batchSize=(example)250000>
```
