import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import *
import torch
import os

def prepare_data(par, t, r, u, u_seq):
   
    # Prepare data according to the different architectures
    if par.architect == 'MLP':
        return MLP_get_dataloader(par, t, r)
    elif par.architect == 'DON' and par.u_delta_fixed:
        return DON_get_dataloader(par, t, r, u, u_seq)
    elif par.architect == 'DON_LH':
        return DON_LH_get_dataloader(par, t, r, u, u_seq)
    elif par.architect == 'LF' and par.u_delta_fixed:
        return LF_get_dataloader(par, t, r, u, u_seq)
    else:
        raise ValueError('The architecture is not implemented yet.')

#######################################################

def MLP_get_dataloader(par, t, r):
    train_dataset = Dataset_MLP(t[0], r[0])
    train_loader = DataLoader(train_dataset, batch_size=par.bs_train, shuffle=False)
    
    return train_loader, train_loader, train_loader


def DON_get_dataloader(par, t, r, u, u_seq):
    t_train, r_train, u_train, u_seq_train, t_val, r_val, u_val, u_seq_val, t_test, r_test, u_test, u_seq_test = _split_sequences_train_test_val(par, t, r, u, u_seq)
    
    train_dataset = DON_prepare_data(par, t_train, r_train, u_train)
    val_dataset = DON_prepare_data(par, t_val, r_val, u_val)
    test_dataset = DON_prepare_data(par, t_test, r_test, u_test, noise_overwrite=0)

    train_loader = DataLoader(train_dataset, batch_size=par.bs_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=par.bs_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=par.bs_test, shuffle=False)

    return train_loader, val_loader, test_loader

def DON_LH_get_dataloader(par, t, r, u, u_seq):
    t_train, r_train, u_train, u_seq_train, t_val, r_val, u_val, u_seq_val, t_test, r_test, u_test, u_seq_test = _split_sequences_train_test_val(par, t, r, u, u_seq)
    
    train_dataset = _DON_LH_prepare_data(par, t_train, r_train, u_train)
    val_dataset = _DON_LH_prepare_data(par, t_val, r_val, u_val)
    test_dataset = _DON_LH_prepare_data(par, t_test, r_test, u_test, noise_overwrite=0)

    train_loader = DataLoader(train_dataset, batch_size=par.bs_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=par.bs_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=par.bs_test, shuffle=False)

    return train_loader, val_loader, test_loader


def LF_get_dataloader(par, t, r, u, u_seq):
    t_train, r_train, u_train, u_seq_train, t_val, r_val, u_val, u_seq_val, t_test, r_test, u_test, u_seq_test = _split_sequences_train_test_val(par, t, r, u, u_seq)

    train_dataset = LF_prepare_data(par, t_train, r_train, u_train, u_seq_train)
    val_dataset = LF_prepare_data(par, t_val, r_val, u_val, u_seq_val)
    test_dataset = LF_prepare_data(par, t_test, r_test, u_test, u_seq_test, noise_overwrite=0)

    train_loader = DataLoader(train_dataset, batch_size=par.bs_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=par.bs_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=par.bs_test, shuffle=False)

    return train_loader, val_loader, test_loader


#######################################################
# Helper functions
#######################################################
def DON_prepare_data(par, t, r, u, noise_overwrite=-1):
    if noise_overwrite != -1:
        state_noise_std = noise_overwrite
    else:
        if par.noise:
            state_noise_std = par.noise_sigma
        else:
            state_noise_std = 0
    
    for i in range(len(t)):
        r[i] = r[i] + torch.normal(mean=0., std=state_noise_std, size=r[i].size())
    
    t_new = torch.zeros_like(t)
    r0 = torch.zeros_like(r)

    N_steps_in_Delta = int(par.u_delta/par.st)
    rem_ind = int(len(t[0])-(np.floor(par.tend/par.u_delta))*N_steps_in_Delta)

    for i in range(len(t)):
        for j in range(int(np.floor(par.tend/par.u_delta))):
            t_shift = j*par.u_delta
            
            #Extract t_shift from this sample t-tshift for the samples i till i+N_steps_in_Delta
            t_new[i,j*N_steps_in_Delta:(j+1)*N_steps_in_Delta,:] = t[i,j*N_steps_in_Delta:(j+1)*N_steps_in_Delta,:] - t_shift
            r0[i,j*N_steps_in_Delta:(j+1)*N_steps_in_Delta,:] = torch.tile(r[i,j*N_steps_in_Delta,:], (N_steps_in_Delta, 1))        
    
        # Fill the last indices 
        if rem_ind>0:
            t_shift = (j+1)*par.u_delta
            t_new[:,(j+1)*N_steps_in_Delta:,:] = t[:,(j+1)*N_steps_in_Delta:,:] - t_shift
            r0[i,(j+1)*N_steps_in_Delta:,:] = torch.tile(r[i,(j+1)*N_steps_in_Delta,:], (rem_ind, 1))

    x1, x2, x3, y = _stack_sequences(par, t_new, u, r0, r)
    dataset = Dataset_DON(x1, x2, x3, y)

    return dataset

 
def LF_prepare_data(par, t, r, u, u_seq, noise_overwrite=-1):
    
    init_state, init_state_noise, time, state, state_noise, control_seq = prep_raw_data(par, t, r, u, u_seq, noise_overwrite)
    datasetRaw = Dataset_LF_raw(init_state, init_state_noise, time, state, state_noise, control_seq)
    dataset = Dataset_LF(datasetRaw, par)

    return dataset


def _DON_LH_prepare_data(par, t, r, u, noise_overwrite=-1):
    if noise_overwrite != -1:
            state_noise_std = noise_overwrite
    else:
        if par.noise:
            state_noise_std = par.noise_sigma
        else:
            state_noise_std = 0
    
    for i in range(len(t)):
        r[i] = r[i] + torch.normal(mean=0., std=state_noise_std, size=r[i].size())
    
    t_LH, u_LH, r0_LH = prep_LH_data(par, t, r, u)
    
    x1, x2, x3, y = _stack_sequences(par, t_LH, u_LH, r0_LH, r)
    
    dataset = Dataset_DON(x1, x2, x3, y)
    return dataset

######################

def prep_LH_data(par, t, r, u):
    t_long_horiz = []
    u_track = []
    r0_track = []

    for i in range(len(t)):   
        count_t = 0
        count_pulse = 0

        u_prev = u[i,0]
        u_pulse = u[i,0]
        r0 = r[i,0]
        t_new = torch.tensor([])
        u_new = torch.tensor([])
        r0_new = torch.tensor([])

        for j in range(len(t[i])):
            if all(u[i,j]==u_prev) and j!=len(t[i])-1: #keep trajectory
                count_t += 1

                if any(u[i,j]!=0):
                    count_pulse += 1
                    u_pulse = u[i,j]

            elif any(u[i,j]!=u_prev) and j!=len(t[i])-1:
                changed_indices = (u[i,j] != u_prev)
                
                if any(u[i,j,changed_indices]!=0): 
                    if count_pulse>0:
                        count_pulse += 1

                    t_this_signal = torch.arange(0, round(count_t*par.st,2), par.st)
                    u_tot = torch.cat((u_pulse,torch.tensor([(count_pulse)*par.st])))
                    u_this_signal = torch.tile(u_tot, (count_t, 1))
                    r0_this_signal = torch.tile(r0, (count_t, 1))

                    t_new = torch.cat((t_new, t_this_signal))   
                    u_new = torch.cat((u_new, u_this_signal))
                    r0_new = torch.cat((r0_new, r0_this_signal))

                    r0 = r[i,j]

                    count_t = 1
                    count_pulse = 0
                else:
                    count_t += 1

                u_prev = u[i,j]

            elif j==len(t[i])-1:
                if count_pulse>0:
                    count_pulse += 1
                t_this_signal = torch.arange(0, round(count_t*par.st+par.st,2), par.st)
                u_tot = torch.cat((u_pulse,torch.tensor([(count_pulse)*par.st])))
                u_this_signal = torch.tile(u_tot, (count_t+1, 1))
                r0_this_signal = torch.tile(r0, (count_t+1, 1))

                t_new = torch.cat((t_new, t_this_signal))
                u_new = torch.cat((u_new, u_this_signal))
                r0_new = torch.cat((r0_new, r0_this_signal))

        u_track.append(u_new)
        r0_track.append(r0_new)
        t_long_horiz.append(t_new)

    t_long_horiz_T = torch.stack(t_long_horiz).unsqueeze(-1)
    u_track_T = torch.stack(u_track)
    r0_track_T = torch.stack(r0_track)

    return t_long_horiz_T, u_track_T, r0_track_T



def prep_raw_data(par, t, r, u, u_seq, noise_overwrite):
    n_control_seq = int(np.ceil((par.tend - par.t0 + par.st) / par.u_delta))
    n_traj = t.shape[0]
    n_t = t.shape[1]

    if noise_overwrite != -1:
        state_noise_std = noise_overwrite
    else:
        if par.noise:
            state_noise_std = par.noise_sigma
        else:
            state_noise_std = 0

    init_state = torch.empty((n_traj, par.N_r))
    init_state_noise = torch.empty((n_traj, par.N_r))
    time = torch.empty((n_traj, n_t, 1))
    state = torch.empty((n_traj, n_t, par.N_r))
    state_noise = torch.empty((n_traj, n_t, par.N_r))
    control_seq = torch.zeros((n_traj, n_control_seq, par.N_r))

    for i in range(len(t)):   
        init_state[i] = r[i,0,:]
        init_state_noise[i] = torch.zeros(par.N_r)
        time[i] = t[i].reshape((-1, 1))
        state[i] = r[i].reshape((-1, par.N_r))
        state_noise[i] = torch.normal(mean=0., std=state_noise_std, size=state[-1].size())
        control_seq[i] = u_seq[i].reshape((-1, par.N_r))

    return init_state, init_state_noise, time, state, state_noise, control_seq



def _split_sequences_train_test_val(par, t, r, u, u_props):
    t_train = t[:int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio))]
    r_train = r[:int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio))]
    u_train = u[:int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio))]
    u_props_train = u_props[:int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio))]

    t_val = t[int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio)):int(par.N_data_seq * (1 - par.test_split_ratio))]
    r_val = r[int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio)):int(par.N_data_seq * (1 - par.test_split_ratio))]
    u_val = u[int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio)):int(par.N_data_seq * (1 - par.test_split_ratio))]
    u_props_val = u_props[int(par.N_data_seq * (1 - par.test_split_ratio - par.val_split_ratio)):int(par.N_data_seq * (1 - par.test_split_ratio))]

    t_test = t[int(par.N_data_seq * (1 - par.test_split_ratio)):]
    r_test = r[int(par.N_data_seq * (1 - par.test_split_ratio)):]
    u_test = u[int(par.N_data_seq * (1 - par.test_split_ratio)):]
    u_props_test = u_props[int(par.N_data_seq * (1 - par.test_split_ratio)):]

    return t_train, r_train, u_train, u_props_train, t_val, r_val, u_val, u_props_val, t_test, r_test, u_test, u_props_test


def _stack_sequences(par, t, u, r0, r):
    N = t.shape[0] #number of sequences
    Nt = t.shape[1] #number of time steps
    
    x1 = torch.reshape(t, (N * Nt, 1))
    x3 = torch.reshape(r0, (N * Nt, par.N_r))
    y = torch.reshape(r, (N * Nt, par.N_r))

    if par.architect == 'DON_LH':
        x2 = torch.reshape(u, (N * Nt, par.N_r+1))
    else:
        x2 = torch.reshape(u, (N * Nt, par.N_r))
    
    return x1, x2, x3, y


######################################################################################################
# Saving and loading the generated datasets

def save_dataloaders(par, train_loader, val_loader, test_loader):
    dataset_name = f"Dataset_{par.architect}_Nr{par.N_r}_Nu{par.N_u}_Nseq{par.N_data_seq}_tend{par.tend}_noise{par.noise}_{par.u_type}"

    if not os.path.exists(par.folder_datasets):
            os.makedirs(par.folder_datasets)
    torch.save(train_loader.dataset, os.path.join(par.folder_datasets, dataset_name + '_TRAIN.pth'))
    torch.save(val_loader.dataset, os.path.join(par.folder_datasets, dataset_name + '_VAL.pth'))
    torch.save(test_loader.dataset, os.path.join(par.folder_datasets, dataset_name + '_TEST.pth'))

    par.save_folder = par.folder_datasets
    par.save_initialisation_settings()

def load_dataloaders(par):
    dataset_name = f"Dataset_{par.architect}_Nr{par.N_r}_Nu{par.N_u}_Nseq{par.N_data_seq}_tend{par.tend}_noise{par.noise}_{par.u_type}"

    train_dataset = torch.load(os.path.join(par.folder_datasets, dataset_name + '_TRAIN.pth'))
    val_dataset = torch.load(os.path.join(par.folder_datasets, dataset_name + '_VAL.pth'))
    test_dataset = torch.load(os.path.join(par.folder_datasets, dataset_name + '_TEST.pth'))

    train_loader = DataLoader(train_dataset, batch_size=par.bs_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=par.bs_val, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=par.bs_test, shuffle=False)

    return train_loader, val_loader, test_loader