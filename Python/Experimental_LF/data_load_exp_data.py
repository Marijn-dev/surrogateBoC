import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import *
import torch
import os
from scipy.io import loadmat

def get_exp_data(par):   
    return LF_get_dataloader(par)


def LF_get_dataloader(par):
    #load and split the data
    t, r, t_seq, u_seq = load_exp_traj(par)
    t_train, r_train, u_seq_train, t_val, r_val, u_seq_val, t_test, r_test, u_seq_test = _split_sequences_train_test_val(par, t, r, u_seq)

    #print the length of t_train with the text: 'length of train set: '
    print('length of train set: ', len(t_train))
    print('length of val set: ', len(t_val))
    print('length of test set: ', len(t_test))

    train_dataset = LF_prep_dataset(par, t_train, r_train, u_seq_train)
    val_dataset = LF_prep_dataset(par, t_val, r_val, u_seq_val)  

    train_loader = DataLoader(train_dataset, batch_size=par.bs_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=par.bs_val, shuffle=False)
    
    return train_loader, val_loader, t_test, r_test, u_seq_test


#######################################################
# Helper functions
#######################################################

def load_exp_traj(par):
    # Load the data
    # Load the .mat file
    data = loadmat('Traj_window_u.mat')

    # Extract cell arrays
    ExpData_t = data['ExpData_t'][0]
    ExpData_fr = data['ExpData_fr'][0]
    ExpData_I_seq = data['ExpData_I_seq'][0]
    ExpData_t_seq = data['ExpData_t_seq'][0]

    # Convert cell arrays to lists of numpy arrays (preserving original 2D shape)
    ExpData_t_list = [np.array(item) for item in ExpData_t]
    ExpData_fr_list = [np.array(item) for item in ExpData_fr]
    ExpData_I_seq_list = [np.array(item) for item in ExpData_I_seq]
    ExpData_t_seq_list = [np.array(item) for item in ExpData_t_seq]

    t = convert_to_tensors(ExpData_t_list)
    r = convert_to_tensors(ExpData_fr_list)
    u_seq = convert_to_tensors(ExpData_I_seq_list)
    t_seq = convert_to_tensors(ExpData_t_seq_list)

    return t, r, t_seq, u_seq 


def _split_sequences_train_test_val(par, t, r, u_props):
    t_train = t[:int(len(t) * (1 - par.test_split_ratio - par.val_split_ratio))]
    r_train = r[:int(len(t) * (1 - par.test_split_ratio - par.val_split_ratio))]
    u_props_train = u_props[:int(len(t) * (1 - par.test_split_ratio - par.val_split_ratio))]

    t_val = t[int(len(t) * (1 - par.test_split_ratio - par.val_split_ratio)):int(len(t) * (1 - par.test_split_ratio))]
    r_val = r[int(len(t) * (1 - par.test_split_ratio - par.val_split_ratio)):int(len(t) * (1 - par.test_split_ratio))]
    u_props_val = u_props[int(len(t) * (1 - par.test_split_ratio - par.val_split_ratio)):int(len(t) * (1 - par.test_split_ratio))]

    t_test = t[int(len(t) * (1 - par.test_split_ratio)):]
    r_test = r[int(len(t) * (1 - par.test_split_ratio)):]
    u_props_test = u_props[int(len(t) * (1 - par.test_split_ratio)):]

    return t_train, r_train, u_props_train, t_val, r_val, u_props_val, t_test, r_test, u_props_test

####


def LF_prep_dataset(par, t, r, u_seq):
    init_state, init_state_noise, time, state, state_noise, control_seq = LF_prep_raw_data(par, t, r, u_seq)
    datasetRaw = Dataset_LF_raw(init_state, init_state_noise, time, state, state_noise, control_seq)
    dataset = Dataset_LF(datasetRaw, par)

    return dataset

####

def LF_prep_raw_data(par, t, r, u_seq):
    
    init_state = []
    init_state_noise = []
    time = []
    state = []
    state_noise = []
    control_seq = []

    for i in range(len(t)):   
        init_state.append(r[i][0,:])
        init_state_noise.append(torch.zeros(par.N_r))
        time.append(t[i])
        state.append(r[i])
        state_noise.append(torch.zeros(size=state[-1].size()))
        control_seq.append(u_seq[i])

    return init_state, init_state_noise, time, state, state_noise, control_seq


####


def convert_to_tensors(numeric_arrays):
    """
    Convert a list of numpy arrays to PyTorch tensors and reshape.
    """
    tensors = []
    for arr in numeric_arrays:
        tensor = torch.tensor(arr, dtype=torch.float32)
        # Reshape tensor to (N, 1) if it was (1, N)
        if tensor.dim() == 2 and tensor.size(0) == 1:
            tensor = tensor.view(-1, 1)
        tensors.append(tensor)
    return tensors