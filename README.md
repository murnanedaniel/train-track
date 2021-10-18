<div align="center">

<figure>
    <img src="https://raw.githubusercontent.com/murnanedaniel/train-track/master/docs/media/logo.png" width="250"/>
</figure>
    
# TrainTrack ML
### Quickly run stages of an ML pipeline from the command line

[Documentation](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/)

[![ci](https://github.com/murnanedaniel/train-track/actions/workflows/ci.yml/badge.svg)](https://github.com/murnanedaniel/train-track/actions/workflows/ci.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


</div>

Welcome to repository and documentation the TrainTrack library. Detailed documentation coming very soon! See [here](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/) for the documentation of the examples of this library. 

## Install

TrainTrack is most easily installed with pip:
```
pip install traintrack
```

## Objective

The aim of TrainTrack is simple: Given any set of self-contained [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) modules, run them in a serial and trackable way. 

At its heart, TrainTrack is nothing more than a loop over the stages defined in a `pipeline.yaml` configuration file. However, it can also handle data processing steps (i.e. non-trainable modules), automatically creates grid scans over combinations of hyperparameters, logs training with (currently) either Tensorboard or Weights & Biases, and can run separate, dependent Slurm batch jobs. It also has an opinionated approach to how data is passed from stage to stage, via Lightning callbacks. In this way, the only code that needs to be written is Lightning modules, all other boilerplate and tracking is handled by TrainTrack. 

## Example

`traintrack` uses two ingredients to run and track your training pipeline: 
1. A project configuration file
2. A pipeline configuration file

It also makes one or two assumptions about the structure of your project. For project `MyFirstMNIST`, we should structure it as
```
📦 MyFirstMNIST
┣ 📂 architectures
┣ 📂 notebooks
┣ 📂 configs
┃ ┣ 📜 project_config.yaml
┃ ┗ 📜 my_first_pipeline.yaml
┗ 📂 logs
```
**Note:** Only `configs/project_config.yaml` is a required file. All else is configurable. An example `project_config.yaml`:
```
# project_config.yaml

# Location of libraries
libraries:
    model_library: architectures
    artifact_library: /my/checkpoint/directory
    

# The lines you would like/need in a batch script before the call to pipeline.py
custom_batch_setup:
    - conda activate my-favorite-environment
    
# If you need to set up some environment before a batch is submitted, define it here in order of commands to run
command_line_setup:
    - module load cuda
    
# If you need to run jobs serially, set to true
serial: False

# Which logger to use - options are Weights & Biases [wandb], TensorBoard [tb], or [None]
logger: wandb
```

We can launch a vanilla run of TrainTrack with 
```
traintrack configs/my_first_pipeline.yaml
```
This trains and performs inference callbacks in the terminal. 


## A Pipeline

The pipeline config file defines a pipeline, for example:
```
# my_first_pipeline.yaml

stages:
    - {set: CNN, name: ResNet50, config: test_train.yaml}

```

which presumes a directory structure of:

```
📦 MyFirstMNIST
┣ 📂 architectures
┃ ┗ 📂 CNN
┃ ┃ ┣ 📜 cnn_base.py
┃ ┃ ┣ 📜 test_train.yaml
┃ ┃ ┗ 📂 Models
┃ ┃ ┃ ┗ 📜 resnet.py

```

Again, see [this repository](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/tree/master/Pipelines/Common_Tracking_Example) for example pipelines in action.

<!-- ## Objectives

1. To abstract away the engineering required to run multiple stages of training and inference with combinations of hyperparameter configurations. [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is used for this, and is a good start, but this library extends Lightning to multiple modules run in series in some dependent way.
2. To present a set of templates, best practices and results gathered from significant trial and error, to speed up the development of others in the domain of machine learning for high energy physics. We focus on applications specific to detector physics, but many tools can be applied to other areas, and these are collected in an application-agnostic way in the [Tools](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/tools/overview/) section.

### Disclaimer:

This repository has been functional, but ugly. It is moving to an "alpha" version which follows many conventions and should be considerably more stable and user-friendly. This transition is expected before May 2021. Please be a little patient if using before then, and if something is broken, pull first to make sure it's not already solved, then post an issue second.

## Intro

To start as quickly as possible, clone the repository, [Install](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart) and follow the steps in [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart). This will get you generating toy tracking data and running inference immediately. Many of the choices of structure will be made clear there. If you already have a particle physics problem in mind, you can apply the [Template](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/choosingguide.md) that is most suitable to your use case.

Once up and running, you may want to consider more complex ML [Models](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/models/overview/). Many of these are built on other libraries (for example [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)).

<div align="center">
<figure>
  <img src="https://raw.githubusercontent.com/HSF-reco-and-software-triggers/Tracking-ML-Exa.TrkX/master/docs/media/application_diagram_1.png" width="600"/>
</figure>
</div>

## Install

It's recommended to start a conda environment before installation:

```
conda create --name exatrkx-tracking python=3.8
conda activate exatrkx-tracking
pip install pip --upgrade
```

If you have a CUDA GPU available, load the toolkit or [install it](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) now. You should check that this is done by running `nvcc --version`. Then, running:

```
python install.py
```

will **attempt** to negotiate a path through the packages required, using `nvcc --version` to automatically find the correct wheels. 

You should be ready for the [Quickstart](https://hsf-reco-and-software-triggers.github.io/Tracking-ML-Exa.TrkX/pipelines/quickstart)!

If this doesn't work, you can step through the process manually:

<table style="border: 1px solid gray; border-collapse: collapse">
<tr style="border-bottom: 1px solid gray">
<th style="border-bottom: 1px solid gray"> CPU </th>
<th style="border-left: 1px solid gray"> GPU </th>
</tr>
<tr>
<td style="border-bottom: 1px solid gray">

1. Run 
`export CUDA=cpu`
    
</td>
<td style="border-left: 1px solid gray">

1a. Find the GPU version cuda XX.X with `nvcc --version`
    
1b. Run `export CUDA=cuXXX`, with `XXX = 92, 101, 102, 110`

</td>
</tr>
<tr style="border-bottom: 1px solid gray">
<td colspan="2">

2. Install Pytorch and dependencies 

```
    pip install --user -r requirements.txt
```

</td>
</tr>
<tr style="border-bottom: 1px solid gray">
<td colspan="2">

3. Install local packages

```pip install -e .```
    
</td>
</tr>
<tr>
<td style="border-bottom: 1px solid gray">

4. Install CPU-optimized packages

```
pip install faiss-cpu
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
``` 
    
    
</td>
<td style="border-left: 1px solid gray">

    
4. Install GPU-optimized packages

```pip install faiss-gpu cupy-cudaXXX```, with `XXX`    

```
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3{Y}_cu{XXX}_pyt{ZZZ}/download.html
```
    
where `{Y}` is the minor version of Python 3.{Y}, `{XXX}` is as above, and `{ZZZ}` is the version of Pytorch {Z.ZZ}.

    e.g. `py36_cu101_pyt170` is Python 3.6, Cuda 10.1, Pytorch 1.70.
   
    
</td>
</tr>
</table>

### Vintage Errors

A very possible error will be
```
OSError: libcudart.so.XX.X: cannot open shared object file: No such file or directory
```
This indicates a mismatch between CUDA versions. Identify the library that called the error, and ensure there are no versions of this library installed in parallel, e.g. from a previous `pip --user` install. -->