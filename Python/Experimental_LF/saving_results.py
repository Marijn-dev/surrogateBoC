import os
import scipy.io as sio
import torch
import numpy as np

def save_single_sim(par,logging,test_results,best_modelNN, f1, f2):
    save_name = f"{par.sim_number}_{par.architect}_PI{par.PI_ML}_Nr{par.N_r}_Nu{par.N_u}_Nseq{par.N_data_seq}"
    
    if par.save_results:
        if not os.path.exists(par.save_folder):
            os.makedirs(par.save_folder)
    par.save_initialisation_settings()
    torch.save(best_modelNN.state_dict(), f"{par.save_folder}/{save_name}_trained_model.pt")
    sio.savemat( f"{par.save_folder}/{save_name}_training_logs.mat", logging)
    sio.savemat( f"{par.save_folder}/{save_name}_test_results.mat", test_results)
    f1.savefig(f'{par.save_folder}/{save_name}_fig_losses.png')
    f2.savefig(f'{par.save_folder}/{save_name}_fig_test_results.png')
    print(f'Finished with saving results and models of this experiment: {par.exp_name}, sim: {par.sim_number}.')


def extract_results_saving(logging, test_results):
    results_sim = {'L_train_NN': logging['loss_train_NN'][logging['best_model_epoch']], 
              'L_train_Phys': logging['loss_train_PHYS'][logging['best_model_epoch']], 
              'L_val_NN': logging['loss_val_NN'][logging['best_model_epoch']], 
              'L_val_Phys': logging['loss_val_PHYS'][logging['best_model_epoch']], 
              'L_test_NN': test_results['loss_data'], 
              'L_test_Phys': test_results['loss_phys'],
              'best_epoch': logging['best_model_epoch'], 
              'stop_epoch': len(logging['loss_train_NN']),
              'train_time': logging['training_time'],
              'phys_gain': logging['gain'][logging['best_model_epoch']]}

    return results_sim


def extract_results_saving_phys_param(par, logging):
    if par.PI_ML:
        est_phys_param = logging['Phys_par'][logging['best_model_epoch']]

        est_alpha = est_phys_param[0]
        est_beta = est_phys_param[1]
        est_gamma = est_phys_param[2]
        est_W = est_phys_param[3]
    else:
        est_alpha = None
        est_beta = None
        est_gamma = None
        est_W = None

    results_phys = {'est_alpha': est_alpha,
                    'est_beta': est_beta,
                    'est_gamma': est_gamma,
                    'est_W': est_W,
                    'true_alpha': par.alpha,
                    'true_beta': par.beta,
                    'true_gamma': par.gamma,
                    'true_W': par.W_flatten}

    for key, value in results_phys.items():
        if isinstance(value, np.ndarray):
            results_phys[key] = value.tolist()
        if isinstance(value, list):
            results_phys[key] = str(value)

    return results_phys


def save_pd_df(par, results_df, df_name='results'):
    save_name = f"{par.architect}_PI{par.PI_ML}_Nr{par.N_r}_Nu{par.N_u}_Nseq{par.N_data_seq}"

    if par.save_results:
        if not os.path.exists(par.save_folder):
            os.makedirs(par.save_folder)

    results_df.to_csv(f"{par.save_folder}/{save_name}_{df_name}.csv")
    par.save_initialisation_settings()
    
    print(f"Results saved as {save_name}_{df_name}.csv")


def save_traj(par, dict_traj, i):
    if par.save_results:
        save_name = f"{par.save_folder}/{par.sim_number}_{par.architect}_traj_{par.u_type}_{i}.mat"

        if not os.path.exists(par.save_folder):
            os.makedirs(par.save_folder)
        
        sio.savemat(save_name, dict_traj)