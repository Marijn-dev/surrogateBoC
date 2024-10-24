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

Different subfolders contain Matlab code. In general, a file name that starts with `fun....m` is a function, and files starting with `main...m` can be run individually, calling on functions in the same folder if needed. The subfolders and tasks executed by the code are briefly explained below:

- **`Analyse_PI_ML_results`**: This subfolder contains code to generate visualizations of the results of the Physics-Informed Machine Learning models.
- **`FHN_experimental_data`**: Learn the Fitzhugh-Nagumo (FHN) model based on experimental data. To run the main file without issues, the `fun_parameters.m` file sets all parameters used in the simulation. The default values can be used, but the directory needs to be set to where the MEA data is stored by adjusting the parameter `pm.loading.measurement` inside this file. The data needs to be in the `.h5` file format.
- **`FHN_numerical_data`**: Learn the Fitzhugh-Nagumo (FHN) model using numerically generated data. The `main` files walk through all steps of setting up the simulation, generating the data, and learning the FHN model parameters.
- **`Prepare_MEA_Measurements`**: This subfolder contains all Matlab code that can be used to analyze the activity in MEA data and to convert the raw data to firing rate model inputs for the physics-informed learning flow function. Just like in the `FHN_experimental_data` subfolder, the `fun_parameters.m` file sets all parameters used in the simulation, and the directory needs to be set to where the MEA data is stored by adjusting the `pm.loading.measurement` parameter inside this file. The raw data needs to be in the `.h5` file format.


---

## Folder: Python

This folder contains all Python code, used for running Physcis-Informed Machine Learning simulations and analyses.

1. **Setting Up the Environment**  
   It is recommended to use a separate environment to avoid package conflicts. The required packages and their versions are listed in the `environment.yaml` file. While you can use this file to set up the environment, manually installing the packages is also an option. Most Python files in this repository require only a few specific packages, which can be easily identified at the top of each script.

2. **Initializing the Simulation**  
   Before running the simulations, configure the settings in `config.py` according to your needs. The default parameters should work for most cases, but ensure you specify the directory where results will be saved by adjusting the `self.save_folder` variable. Similarly, when saving datasets, update the storage path by setting `self.folder_datasets`.

3.  **Run the simulation**
    Once everything is set, run any of the `main...py` scripts to start the simulation. The `main_single.py` is running a single loop of data generation, learning the model and testing the model. All other `main...py` files are more extensive numerical experiments. The results will be automatically stored in the specified directory. To generate the figures as found in the thesis connected to this repository, use the MATLAB scripts found in the `Analyse_PI_ML_results` subfolder inside the `Matlab` folder of this repository.
