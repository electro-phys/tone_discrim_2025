# tone_discrim_2025
Contains code for associated spiking data processing used in Gauthier et al. 2025


## Rate Level Function Fitting
This code can be ran as is. All functions are self contained in the script. Spiking data is included in both an example spiking dataset for trouble shooting as well as the full Auditory Cortex and Inferior Colliculus datasets (acx_spks_full.pkl and ic_spks_full.pkl) respectively.


## Tuning Quantification
This code contains functionality for quantifying tuning properties via SPIKE-distance [SPIKE-distance](http://www.scholarpedia.org/article/SPIKE-distance) from the same spiking data. Note that this code requires dataframe of characteristic frequency and thresholds for each unit. These are provided in the repository. This code requries installation of SPIKE-distance functionality from the [PySpike](https://github.com/mariomulansky/PySpike) repository. Note that this requires setting up a C-based compiler for python [Cython](https://cython.org/) to run.
