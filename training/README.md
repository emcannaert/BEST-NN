# Training the Boosted Event Shape Tagger

The programs in this directory train the BEST.

### Test Code Directory

The ``test`` directory contains scripts used to test different Scalers and Neural Net Archetectures for BEST. These are not normally necessary to train BEST, but could be of use to those who would like to modify BEST.

## Using the FermiLab GPUs

BEST is a complicated network that takes a lot of processing to train, so we recommend training it using a GPU. 
To use the GPU at FermiLab, set up an appropriate GPU environment. **Do NOT use the CMS environment.**
[Learn more about the cmslpc gpus.](https://uscms.org/uscms_at_work/computing/setup/gpu.shtml)
First, log into one of the three gpu nodes (cmslpcgpu1, cmslpcgpu2, or cmslpcgpu3). Then, set up a screen so your tests can safely run for a long time.

```bash
ssh -Y USERNAME@cmslpcgpu3.fnal.gov
screen
bash -l
```

### mlenv0 Environment 

The Basic Neural Net version of BEST runs on tensorflow 1.12. The (depreciated) mlenv0 environment is used to run this version of BEST. To set up: 

```bash
source /cvmfs/cms-lpc.opensciencegrid.org/sl7/gpu/Setup.sh
source activate mlenv0
```

### Singularity Containers
The Point Cloud Transformer (PCT) runs on tensorflow 2.6. To set up a Docker environment with Singularity: 

```bash
singularity run --nv --bind `readlink $HOME` --bind `readlink -f ${HOME}/nobackup/` --bind /uscms_data/d3/bonillaj/ --bind /cvmfs /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:tensorflow-latest-gpu-singularity
```

You must bind any directories that you want to access from within the singularity container. Here, ``/uscms_data/d3/bonillaj/`` is where our datasets are stored. There seems to be a bug with accessing the nobackup spaces, as the symbolic link uses ``uscms_data/d1/``. Using ``uscms_data/d3/`` to access the nobackup space seems to work for now. 

**Do not bind your CWD.** It is recommended to make a symbolic link from your home directory to your working BEST directory, which will allow you to access your CWD from the virtual environment.

```bash
cd ~
ln -s /uscms_data/d3/username/PATH_TO_CMSSW/CMSSW_10_6_27/src/BEST best
```

#### Version Control
In case this Singularity Docker container changes names or is updated in the future, check the [fnallpc Docker Hub](https://hub.docker.com/r/fnallpc/fnallpc-docker/tags), where we use [this tag](https://hub.docker.com/layers/fnallpc-docker/fnallpc/fnallpc-docker/tensorflow-latest-gpu-singularity/images/sha256-708a3151db6c051f2ba88ed0497a5545ef9062a2b56f7e35583c45818c807fc6?context=explore) specifically. According to the tag page, this command will add the image if it is no longer available:

```bash
ADD file:e729fb032bd2f7cde20fb343da0cd358447e8b23028422c123944e8d0be660fa in / 
```

## Training Basic Neural Net BEST with Shape-Matching

The pT of the samples have been decorrelated (shape-matched) in the formatConverter step. Before the training, one must standardize the inputs for all file sets. Then one trains and evaluates the network. There are two architectures available, oldBEST and nnBEST. These are fairly simple architectures, and can be used to test the BEST Input Variables before using the much more complex PCT. Options are available to choose the input data. Make sure that your mlenv0 environment is set up prior to standardizing or training BEST. If your h5 input sample files differ, you will need to update the lists "sampleTypes" and "samples" that appear throughout this directory and others.

### Standardize Input Variables
```bash
# Options can be specified to tweak standardization. Use help flag for more information:
python MakeStandardInputs.py --help
```

### Train oldBEST
```bash
# Many options can be specified to tweak training. Use help flag for more information:
python oldBEST.py --help
```

### Train nnBEST
```bash
# Many options can be specified to tweak training. Use help flag for more information:
python nnBEST.py --help
```

### Plotting Performance

The performance of the models is plotted by the ``plotAll`` function within ``tools/functions.py``. The training scripts above will automatically call this plotting script. However, to plot the performance of an already existing, already trained model, one can add the ``-t`` flag to either training script above. This information is also available in the help message for those training scripts.

### Plotting Input Variables

The BEST input variables can be plotted from their respective h5 files using ``plotBESTInputs.py``. This is helpful in debugging potential issues with the input data.



## Warning About Functions in Python

Python does not forget about operations done to a variable inside a function. If a variable ``var`` is declared
in the main program and a function then deletes ``var`` in order to return something else, ``var`` will also be
deleted from the main program. This also includes any variables that point to the same memory; for example 
``var2 = var`` will also be deleted. To avoid this, use the copy module to copy the memory.

```python
import copy
var2 = copy.copy(var)
result = function(var) # function that deletes var
# var2 will still be here, but var will be deleted
```

