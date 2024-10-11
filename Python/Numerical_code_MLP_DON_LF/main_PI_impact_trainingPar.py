import os
import scipy.io as sio
import sys
import pandas as pd

from config import Config
from data_preparation import load_dataloaders
from model import *
from train import train_model
from testing import test_model
from saving_results import *

def main():
    # Initialize settings
    config = Config()
    n_sim_per_combi = 10

    # Load dataloaders
    train_loader, val_loader, test_loader = load_dataloaders(config)

    param_combi = [
    {'PI_ML': False, 'eta_gain': 0.01, 'init_gain': 0},
    {'PI_ML': True, 'eta_gain': 0.001, 'init_gain': 0},
    {'PI_ML': True, 'eta_gain': 0.005, 'init_gain': 0},
    {'PI_ML': True, 'eta_gain': 0.01, 'init_gain': 0},
    {'PI_ML': True, 'eta_gain': 0.05, 'init_gain': 0},
    {'PI_ML': True, 'eta_gain': 0.1, 'init_gain': 0},
    {'PI_ML': True, 'eta_gain': 0.001, 'init_gain': 0.1},
    {'PI_ML': True, 'eta_gain': 0.005, 'init_gain': 0.1},
    {'PI_ML': True, 'eta_gain': 0.01, 'init_gain': 0.1},
    {'PI_ML': True, 'eta_gain': 0.05, 'init_gain': 0.1},
    {'PI_ML': True, 'eta_gain': 0.1, 'init_gain': 0.1},
    {'PI_ML': True, 'eta_gain': 0.001, 'init_gain': 0.5},
    {'PI_ML': True, 'eta_gain': 0.005, 'init_gain': 0.5},
    {'PI_ML': True, 'eta_gain': 0.01, 'init_gain': 0.5},
    {'PI_ML': True, 'eta_gain': 0.05, 'init_gain': 0.5},
    {'PI_ML': True, 'eta_gain': 0.1, 'init_gain': 0.5}]

    all_results = []
    all_results_phys = []

    #run per different combi of parameters n_sim_per_combi simulations
    for i, combi in enumerate(param_combi):
        for key in combi:
            setattr(config, key, combi[key])

        for n_sim in range(n_sim_per_combi):
            result, result_phys = run_simulation(config, train_loader, val_loader, test_loader)
            result['combi'] = i
            result.update(combi)
            all_results.append(result)
            all_results_phys.append(result_phys)

            #print statement updating which simulation is done
            print(f"\n _____________________________________________________ \n")
            print(f"Simulation {n_sim} of {n_sim_per_combi} for {combi} done.")
            print(f"\n _____________________________________________________ \n")
            
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    results_phys_df = pd.DataFrame(all_results_phys)

    # Display the DataFrame
    print(results_df.head())
            
    if config.save_results:
        save_pd_df(config, results_df)
        save_pd_df(config, results_phys_df, df_name='phys_par')


def run_simulation(config, train_loader, val_loader, test_loader):
    # Initialize model	
    modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop = initialize_model(config)

    # Train model
    logging, best_modelNN, best_modelPhys = train_model(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, config)    
    
    # Test model
    test_results = test_model(best_modelNN, best_modelPhys, test_loader, criterion, config)
    
    result = extract_results_saving(logging, test_results)
    result_phys = extract_results_saving_phys_param(config, logging)

    return result, result_phys


if __name__ == "__main__":
    main()