import os
import scipy.io as sio
import sys
import pandas as pd

from config import Config
from data_generation import generate_data
from data_preparation import prepare_data
from model import *
from train import train_model
from testing import test_model
from saving_results import *
import wandb

def main():
    # Initialize all settings
    config = Config()

    n_sim_per_combi = 2
    
    param_combi = [{'N_r': 30, 'N_u': 10},
                   {'N_r': 120, 'N_u':30},
                   {'N_r':240, 'N_u':60}]

    all_results = []
    all_results_phys = []

    #run per different combi of parameters n_sim_per_combi simulations
    for i, combi in enumerate(param_combi):
        for key in combi:
            setattr(config, key, combi[key])
        config.update_phys_par()

        #Generate and prepare data
        t, r, u, u_props = generate_data(config, config.N_data_seq)
        train_loader, val_loader, test_loader = prepare_data(config, t, r, u, u_props)

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
        save_pd_df(config,results_df)
        save_pd_df(config, results_phys_df, df_name='phys_par')


def run_simulation(config, train_loader, val_loader, test_loader):
    # Initialize model	
    modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop = initialize_model(config)

    if config.save_results: 
        wandb.init(project='BoC',name=f"{config.exp_name}_Sim:{config.sim_number}_Nr:{config.N_r}_Nu:{config.N_u}",config=config)
        
    # Train model
    logging, best_modelNN, best_modelPhys = train_model(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, config)    
    

    # Test model
    test_results = test_model(best_modelNN, best_modelPhys, test_loader, criterion, config)
    
    result = extract_results_saving(logging, test_results)
    result_phys = extract_results_saving_phys_param(config, logging)

    wandb.finish()
    return result, result_phys


if __name__ == "__main__":
    main()