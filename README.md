# **DFMC: Dead Fuel Moisture Content model**

The model simulates moisture content of dead fuel, with sub-daily resolution and fuel-specific parameters.

This repository provides code and data for the model's definition and calibration process descibed in: 

        Perello et al., "An Hourly Fuel-specific Dead Fuel Moisture Content Model for Wildfire Danger Assessment" (submitted)

See *interactive_run.ipynb* for an example of how to run the model.

The codes of the calibration framework presented in the paper are listed in the *calibration/* folder. The framework can be used with different dataset by adapting the configuration file - see the *configuration.json.sample*. 

The model has been calibrated with hourly 10-hours fuel sticks measurements. Two different datasets have been used for calibration and validation:
* **ISR_dataset**: the original data has been kindly shared for research purpose by the authors of (1). The original dataset has been randomly sampled as described in the paper. While the original dataset could be available under reasonable request to its owners, in this repository just the post-processed data used for calibration and validation is present;
* **BC_dataset**: the original data has been share by the author of (2) through the GitHub repository (https://github.com/dvdkamp/fsmm?tab=readme-ov-file). The dataset has been used for validation.

**References**
1. Shmuel, A., Ziv, Y., Heifetz, E., 2022. Machine-Learning-based evaluation of the time-lagged effect of meteorological factors on 10-hour dead fuel moisture content. Forest Ecology and Management 505, 119897. doi:10. 1016/j.foreco.2021.119897
2. van der Kamp, D.W., Moore, R.D., McKendry, I.G., 2017. A model for simulating the moisture content of standardized fuel sticks of various sizes. Agricultural and Forest Meteorology 236, 123–134. doi:10.1016/j.agrformet.2017.01.013
