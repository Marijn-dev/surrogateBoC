import os
import scipy.io as sio

from config import Config
from data_generation import generate_data
from data_preparation import prepare_data
from model import *
from train import train_model
from testing import test_model
from visualization import plot_training, plot_results
from saving_results import save_single_sim
import wandb

def main():
    # Initialize all settings
    config = Config()
 
    # start Weights and Bias logging
    if config.save_results:
        wandb.init(project='BoC',name=f"{config.exp_name}_{config.sim_number}",config=config)

    # Generate data
    t, r, u, u_props = generate_data(config, config.N_data_seq)
         
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config, t, r, u, u_props)

    # Initialize model	
    modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop = initialize_model(config)

    # Train model
    logging, best_modelNN, best_modelPhys = train_model(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, config)    
     
    # Test model
    test_results = test_model(best_modelNN, best_modelPhys, test_loader, criterion, config)
    
    # Plot results
    f1 = plot_training(logging, config)
    f2 = plot_results(test_results, config)     # plt.show(block=False)

    if config.save_results:
        save_single_sim(config,logging,test_results,best_modelNN, f1, f2)

if __name__ == "__main__":
    main()