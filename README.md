# Read me on the surrogate model for a BoC device

This repository contains three folders. One folder containing all the Matlab code, one folder for the python code and a folder on the Multi-Electrode-Array (MEA) data
The content of these folders and how to set up and use the code in this repository is described folder by folder below. 

---

## Folder: MEA_data
The project data is too large to be stored directly on GitHub. However, the data remains accessible through **PhD Guillem Monso Garcia**.

### Where to Find the Data
To access the MEA data used in this project measured on 02-07-2024, permission needs to be asked for the path in Guillem's database:
9_Experiments -> Experiment Execution -> iNeurons -> D42

### Information on the measurements in this repository
In this repository an excel file is available that describe all the different stimulation characteristics that are applied for each measurement. This file is called '0209_log_measurement.xlsx'.
Additionally, some images are visible from device A. 

### Converting the Data
To convert the data, you will need the **Multichannel System Datamanager** software tool.
With this you are able to convert the raw data to '.h5' format which enable loading in the data in Matlab
You can download the software and follow the installation instructions from the official website:
[Multichannel Systems - Multi Channel Datamanager](https://www.multichannelsystems.com/software/multi-channel-datamanager)

---

## Folder: Matlab 
Different subfolders are found that contain Matlab code. In general a file name that starts with 'fun....m' is a function and files starting with 'main...m' can be run individually, calling on functions in the same folder if needed. The subfolders and tasks executed by the code are shortly elaborated below: 
- **'Analyse_PI_ML_results'**: This subfolder contains code to generate visualizations of the results of the Physics-Informed Machine Learning models. 
- **'FHN_experimental_data'**: Learn the Fitzhugh-Nagumo (FHN) model based on experimental data. To run the main file without issues, the 'fun_parameters.m' sets all parameters used in the simulation. The default values can be used, but the directory needs to be set to where the MEA data is stored by adjusting parameter 'pm.loading.measurement' inside this file. The data needs to be of the format '.h5' file.
- **'FHN_numerical_data'**: Learn the Fitzhugh-Nagumo (FHN) model based by using numerically generated data. The 'main' files run through all steps of setting up the simulation, generating the data and learning the FHN model parameters. 
- **'Prepare_MEA_Measurements'**: This subfolder contains all Matlab code that can be used to analyze the activity in the MEA data and to convert the raw data to firing rate model inputs for the physics informed learning flow function. Just as for the 'FHN_experimental_data', the 'fun_parameters.m' sets all parameters used in the simulation and the directory needs to be set to where the MEA data is stored by adjusting parameter 'pm.loading.measurement' inside this file. The raw data needs to be of the format '.h5' file.

---

## Folder: Python 
text

### Before running the code
1. Setting up the environment

2. Initializing the simulation. 



### Running the code

---