import os
import scipy.io as sio
import sys
import pandas as pd

from config import Config
from data_preparation import load_dataloaders, prepare_data
from data_generation import generate_data
from model import *
from train import train_model
from testing import test_model, test_single_sample
from saving_results import *
from visualization import plot_training, plot_results

def main():
    # Initialize all settings
    config = Config()
    n_traj_save = 5 # indicate how many trajectories should be saved.

    train_loader, val_loader, test_loader = load_dataloaders(config)
    
    # Initialize model	
    modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop = initialize_model(config)

    # Train model
    logging, best_modelNN, best_modelPhys = train_model(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, config)    

    test_results = test_model(best_modelNN, best_modelPhys, test_loader, criterion, config)

    f1 = plot_training(logging, config)
    f2 = plot_results(test_results, config) 

    # Test model on input distribution as training data
    l_data_log = []
    l_phys_log = []
    for i, (sample) in enumerate(test_loader):
        data_loss, phys_loss, traj_log = test_single_sample(modelNN, modelPhys, criterion, sample, config)
        l_data_log.append(data_loss)
        l_phys_log.append(phys_loss)

        #Store trajectory data
        if i < n_traj_save:
            save_traj(config, traj_log, i)

    # Make testloader with sinusoidal input distribution
    config.u_type = 'sinusoidal'
    _, _, test_loader_sin = load_dataloaders(config)
    

    # Test model on sinusoidal distribution as training data
    l_data_log_sin = []
    l_phys_log_sin = []
    for i, (sample) in enumerate(test_loader_sin):
        data_loss, phys_loss, traj_log = test_single_sample(modelNN, modelPhys, criterion, sample, config)
        l_data_log_sin.append(data_loss)
        l_phys_log_sin.append(phys_loss)

        #Store trajectory data
        if i < n_traj_save:
            save_traj(config, traj_log, i)

    test_samples_log = {'l_data': l_data_log, 'l_phys': l_phys_log, 'l_data_sin': l_data_log_sin, 'l_phys_sin': l_phys_log_sin}
    df_test_samples_log = pd.DataFrame(test_samples_log)

    if config.save_results:
        save_single_sim(config,logging,test_results,best_modelNN, f1, f2)
        save_pd_df(config,df_test_samples_log,'test_samples')


if __name__ == "__main__":
    main()