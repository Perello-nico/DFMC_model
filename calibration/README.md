# Model Calibration Framework
The folder contains the codes of the calibration framework presented in the paper. The framework can be used to calibrate different parameters - see the *configuration.json.sample* for an example of the configuration file.

Also, it can be applied to different time series dataset, by adding the *data_path_calib* and *data_path_valid* options in the configuration file - see the *configuration_new-data.json.sample* for an example. If not provided, the default dataset used in paper will be adopted.
**NOTE**: the provided data has to be compliant with the structure of the dataset as presented in the *data.sample.pkl* file.


### Calibration data 
The data used to test the calibration framework and to calibrate the model for the paper are available in the *data_calibration/* folder.
The model has been calibrated with hourly 10-hours fuel sticks measurements. Two different datasets have been used for calibration and validation:
* **ISR_dataset**: the original data has been kindly shared for research purpose by the authors of (1). The original dataset has been randomly sampled as described in the paper. While the original dataset could be available under reasonable request to its owners, in this repository the processed data used for calibration and validation is present i.e., a sampling of time series as described in the paper. This dataset have been used both for **calibration** and **validation**;
* **BC_dataset**: the original data has been share by the author of (2) through the GitHub repository (https://github.com/dvdkamp/fsmm?tab=readme-ov-file). The dataset has been used for **validation**.

The results of the calibration process are available in the *results_calibration/* folder.


**References**
1. Shmuel, A., Ziv, Y., Heifetz, E., 2022. Machine-Learning-based evaluation of the time-lagged effect of meteorological factors on 10-hour dead fuel moisture content. Forest Ecology and Management 505, 119897. doi:10. 1016/j.foreco.2021.119897
2. van der Kamp, D.W., Moore, R.D., McKendry, I.G., 2017. A model for simulating the moisture content of standardized fuel sticks of various sizes. Agricultural and Forest Meteorology 236, 123–134. doi:10.1016/j.agrformet.2017.01.013
