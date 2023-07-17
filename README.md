![Build](https://github.com/maciejzj/cnn-res-degrader/workflows/Build/badge.svg)

# CNN resolution degrader

Convolutional neural networks for downscaling aerial images. Trained on a set
of real-life high and low-resolution images from the Proba-V satellite mission.
Created as a part of data augmentation and generation for super-resolution
project.

This is part of a research project published at:

M. Ziaja, J. Nalepa and M. Kawulok, "Data Augmentation for Multi-Image 
Super-Resolution," IGARSS 2022 - 2022 IEEE International Geoscience and 
Remote Sensing Symposium, Kuala Lumpur, Malaysia, 2022, pp. 119-122,
[doi: 10.1109/IGARSS46834.2022.9884609](https://ieeexplore.ieee.org/document/9884609).

## Running the project

The data is managed mostly by the [DVC](https://dvc.org) system. To obtain
access to the DVC remotes please contact the maintainer: `maciejzjg@gmail.com`.

There are two ways of managing data, one can either use the full Proba-V dataset
for degradation training and then create *artificial* Sentinel-2 datasets for
super-resolution, or split Proba-V in two and use one half for degradation
fitting and the other for super-resolution. These splits of Proba-V are denoted
as *a* and *b* consecutively.

### Input datasets

The input dataset generation for training degradation networks is stored in the
`data_generation_degradation` (may require some readjustments to regenerate the
datasets). This part of data preprocessing is not managed by DVC pipelines,
although the datasets are stored with DVC. They can be downloaded from remote
with the `dvc pull` command. 

### Networks training

The models are automatically managed depending on changes in source code and
configuration in `params.yaml`. The training pipeline is defined in `dvc.yaml`.
Experiment reproduction is done with the `dvc repro` command. Pre-trained
models created with the current state of the `master` branch can be pulled from
the DVC remote.  There are three model architectures for degradation: a simple
fully convolutional network, a U-Net, and a GAN network.

### Evaluation

Evaluation is handled outside of the DVC pipelining. To test run the 
`python -m cnn_res_degrader.test` (use help to get familiar with the usage). The
results are presented as heatmap plots. The GAN network doesn't feature early
stopping and checkpointing best model, so every epoch the generator network
is saved. To examine which epoch performed best on the validation subset use the
`find_best_gan.py` script.

### Exporting/data generation

*Artificial* datasets for super-resolution training can be generated using
scripts from the `data_generation_superres`. Consult the help for usage.

## Misc

Some insights and data exploration can be found in Jupyter notebooks in the
`analysis` directory.

The results of super-resolution trainings using artificial datasets and a wider
context can be found in a paper in
[this](https://github.com/maciejzj/masters-thesis) repository (however, it may
be outdated). The work is continued at [KP Labs](https://kplabs.space).
