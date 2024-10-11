import os
import scipy.io as sio
import pandas as pd
from torch.utils.data import DataLoader

#Self written functions
from config import Config
from data_load_exp_data import get_exp_data, LF_prep_dataset
from model import *
from train import train_model
from testing import test_single_sample
from visualization import plot_training
from saving_results import save_single_sim, save_traj, save_pd_df

def main():
    # Initialize all settings
    config = Config()

    # Get experimental data
    train_loader, val_loader, t_test, r_test, u_seq_test  = get_exp_data(config)

    # Initialize model	
    modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop = initialize_model(config)

    # Train model
    logging, best_modelNN, best_modelPhys = train_model(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, config)    
    f1 = plot_training(logging, config)

    if config.save_results:
        save_single_sim(config,logging,{},best_modelNN, f1, f1)

    # Test model on input distribution as training data
    l_data_log = []
    l_phys_log = []

    for k in range(len(t_test)):
        test_dataset = LF_prep_dataset(config, [t_test[k]], [r_test[k]], [u_seq_test[k]])
        test_loader = DataLoader(test_dataset, batch_size=t_test[k].shape[0], shuffle=False)

        for i, (sample) in enumerate(test_loader):
            data_loss, phys_loss, traj_log = test_single_sample(modelNN, modelPhys, criterion, sample, config)
            l_data_log.append(data_loss)
            l_phys_log.append(phys_loss)

            #Store trajectory data
            if config.save_results:
                save_traj(config, traj_log, k)

    test_samples_log = {'l_data': l_data_log, 'l_phys': l_phys_log}
    df_test_samples_log = pd.DataFrame(test_samples_log)

    if config.save_results:
        save_pd_df(config,df_test_samples_log,'test_samples')

if __name__ == "__main__":
    main()