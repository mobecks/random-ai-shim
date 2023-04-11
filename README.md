# Acquisitions with random shim values enhance AI-driven NMR shimming

This is the official project page including the [paper](doi.org/10.1016/j.jmr.2022.107323), code, models and a link to the dataset.

## Description

Shimming is still an unavoidable, time-consuming and cumbersome burden that precedes NMR
experiments, and aims to achieve a homogeneous magnetic field distribution, which is required for
expressive spectroscopy measurements. This study presents multiple enhancements to AI-driven
shimming.We achieve fast, quasi-iterative shimming on multiple shims simultaneously via a temporal
history that combines spectra and past shim actions. Moreover, we enable efficient data collection by
randomized dataset acquisition, allowing scalability to higher-order shims. Application at a low-field
benchtop magnet reduces the linewidth in 87 of 100 random distortions from ~ 4 Hz to below 1 Hz,
within less than 10 NMR acquisitions. Compared to, and combined with, traditional methods, we
significantly enhance both the speed and performance of shimming algorithms. In particular, AIdriven
shimming needs roughly 1/3 acquisitions, and helps to avoid local minima in 96% of the cases.
Our dataset and code is publicly available.

## Paper

Paper published at Journal of Magnetic Resonance: [doi.org/10.1016/j.jmr.2022.107323](doi.org/10.1016/j.jmr.2022.107323).

M.Becker, S.Lehmkuhl, S.Kesselheim, J.G.Korvink, M.Jouda, Acquisitions with random shim values enhance AI-driven NMR shimming, 2022 

## RandomShimDB 

The shimming database ShimDB, a collection of proton NMR signals recorded under the application of shim coil fields, has been extended with our random dataset.

For more information and downloading of subsets of ShimDB, see [this page](https://github.com/mobecks/ShimDB).

## Execution

Scripts contain a dictionary variable ```initial_config```, which allows for easy modifications, e.g. to choose hyperparameters.

### Offline DL training

Requirements: RandomShimDB dataset.

First, specify all required paths in ```eDR_Training.py``` (Tip: search for "TODO").
Then, run ```$python eDR_Training.py``` to train the convolutional LSTM architecture. Start a limited hyperparameter optimization run with the argument ```--raytuning 1```.

### In-situ evaluation

Requirements: Trained neural network as .pt-file, NMR spectrometer.

Experiments are conducted on a Spinsolve 80 spectrometer (Magritek GmbH, Aachen, Germany, www.magritek.com) with an interface to the Spinsolve Expert software. Scripts that need to be executed on the machine are located in the folder "MagritecScripts/".

Run ```$python eDR_deployment.py``` on the PC connected to your Magritek device to start the evaluation protocol.
Run ```$python eDR_comparison_*.py``` to start the comparison routines to either parabola or simplex shimming.

**Note**: Due to licence restrictions, specific files for communication to the spectrometer and triggering the experiments may be missing. This includes custom scripts in the Spinsolve Expert software that can receive external variables from python.


## Dependencies
Code was developed and tested with the following packages: 

- conda v4.13 (including all default packages) [Installation guide](https://docs.anaconda.com/anaconda/install/index.html)
- python v3.8.11
- pytorch v1.9.1 [Installation guide](https://pytorch.org/get-started/locally/)
- nmrglue v0.9.dev0 [Github](https://github.com/jjhelmus/nmrglue)
- ray v1.6.0 (+ ray tune) [Website](https://docs.ray.io/en/latest/tune/index.html)
- scienceplots v1.0.9 [Github](https://github.com/garrettj403/SciencePlots)


## Citation

If you find this method useful and want to cite it, please use the following bibtex entry:

```
@article{BECKER2022AIshimming,
title = {Acquisitions with random shim values enhance AI-driven NMR shimming},
author = {Moritz Becker and Sören Lehmkuhl and Stefan Kesselheim and Jan G. Korvink and Mazin Jouda},
journal = {Journal of Magnetic Resonance},
pages = {107323},
year = {2022},
issn = {1090-7807},
doi = {https://doi.org/10.1016/j.jmr.2022.107323},
}
```
