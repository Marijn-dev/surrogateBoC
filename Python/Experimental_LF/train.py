import os
import torch
from tqdm import tqdm
import numpy as np
import time

##################################################################################
 # General training function
##################################################################################
def train_model(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, par, overwrite_arch = ''):
    
    print("Start training learning flow model")
    return train_LF(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, par)
    
    

###################################################################################
# Training functions for different architectures
##################################################################################



def train_LF(modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop, train_loader, val_loader, par):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_loss = float('inf')
    min_loss_data = float('inf')
    logging = {'loss_train_NN': [], 'loss_train_PHYS': [], 'loss_train_tot': [],
               'loss_val_NN': [], 'loss_val_PHYS': [], 'loss_val_tot': [], 
               'gain': [], 'Phys_par': [], 'best_model_epoch': 0, 'training_time': 0}
    
    start_time = time.time()

    for epoch in tqdm(range(par.num_epochs), miniters=50, mininterval=10, maxinterval=60):
        #### Training
        modelNN.train()
        modelPhys.train()
        train_loss_NN = 0.0
        train_loss_PHYS = 0.0
        
        for i, (sample) in enumerate(train_loader):  
            
            r0, r_batch, u_inn, deltas, deltasPhys, u_inst, t, delta_window = lf_prep_inputs(*sample, device, par)
            
            optimizer.zero_grad()
            optimizer_gain.zero_grad()
        
            # Data Loss 
            r_data = modelNN(r0, u_inn, deltas)
            loss_data = criterion(r_data, r_batch)
            
            if par.PI_ML: # WITH PHYS
                # Physics Loss 
                r_phys = modelNN(r0, u_inn, deltasPhys)
                output_phys = modelPhys(r_phys, u_inst, delta_window)

                grads = []
                for i in range(par.N_r):
                    gradient_r = torch.autograd.grad(r_phys[:, i], deltasPhys, torch.ones_like(r_phys[:, i]), create_graph=True)[0]
                    gradient_r = torch.sum(gradient_r, (-2,-1))*(1/par.u_delta)
                    grads.append(gradient_r)
                dr_phys_dt = torch.stack(grads, dim=1)

                loss_phys = criterion(dr_phys_dt-output_phys,torch.zeros_like(dr_phys_dt))
                loss = loss_data+gain*loss_phys    
            
            else: # WITHOUT PHYS
                loss = loss_data
            
            loss.backward()
            optimizer.step()

            train_loss_NN += loss_data.item()
            if par.PI_ML:
                train_loss_PHYS += loss_phys.item()
            else:
                train_loss_PHYS = 0.0

            if epoch>par.start_gain_update and par.PI_ML:
                if (min_loss_data-loss_data.item()) > - 1e-4: #update gain with improvement or roughly the same loss
                    optimizer_gain.step() 
                    min_loss_data = loss_data.item()

        train_loss_NN /= len(train_loader)
        train_loss_PHYS /= len(train_loader)

        #### Validation
        modelNN.eval()
        modelPhys.eval()
        val_loss_NN = 0.0
        val_loss_PHYS = 0.0

        for i, (sample) in enumerate(val_loader):  
            r0, r_batch, u_inn, deltas, deltasPhys, u_inst, t, delta_window = lf_prep_inputs(*sample, device, par)

            with torch.no_grad():
                r_data = modelNN(r0, u_inn, deltas)
                loss_data = criterion(r_data, r_batch)
            
            if par.PI_ML:     
                r_phys = modelNN(r0, u_inn, deltasPhys)
                output_phys = modelPhys(r_phys, u_inst, delta_window)

                grads = []
                for i in range(par.N_r):
                    gradient_r = torch.autograd.grad(r_phys[:, i], deltasPhys, torch.ones_like(r_phys[:, i]), create_graph=True)[0]
                    gradient_r = torch.sum(gradient_r, (-2,-1))*(1/par.u_delta)
                    grads.append(gradient_r)
                dr_phys_dt = torch.stack(grads, dim=1)

                loss_phys = criterion(dr_phys_dt-output_phys,torch.zeros_like(dr_phys_dt))
                loss = loss_data+gain*loss_phys
            else:
                loss = loss_data

            val_loss_NN += loss_data.item()
            if par.PI_ML:
                val_loss_PHYS += loss_phys.item()
            else:
                val_loss_PHYS = 0.0

        val_loss_NN /= len(val_loader)
        val_loss_PHYS /= len(val_loader)
        
        sched.step((val_loss_NN+gain.item()*val_loss_PHYS))
        early_stop.step(val_loss_NN, val_loss_PHYS)

        logging = log_progress(logging, epoch, train_loss_NN, train_loss_PHYS, val_loss_NN, val_loss_PHYS, gain, modelPhys.parameters(), par)

        if (val_loss_NN+gain.item()*val_loss_PHYS) < best_val_loss:
            best_modelNN = modelNN
            best_modelPhys = modelPhys
            logging['best_model_epoch'] = epoch
            best_val_loss = (val_loss_NN+gain.item()*val_loss_PHYS)

        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    logging['training_time'] = time.time() - start_time
    
    return logging, best_modelNN, best_modelPhys

###################################################################################
# Helper functions
###################################################################################

def lf_prep_inputs(x0, y, u, lengths, t, device, par):
    sort_idxs = torch.argsort(lengths, descending=True)
    
    x0 = x0[sort_idxs]
    y = y[sort_idxs]
    u = u[sort_idxs]
    lengths = lengths[sort_idxs]
    t = t[sort_idxs]

    deltas = u[:, :lengths[0], -1].unsqueeze(-1)

    deltasPhys = deltas.clone()
    u_instant = torch.zeros(y.shape[0],y.shape[1]+3)
    delta_window = torch.zeros(y.shape[0],1)

    if par.PI_ML:
        for i in range(deltas.shape[0]):
            delta_window[i] = torch.rand(1)
            deltasPhys[i, lengths[i].item()-1] = delta_window[i]
            u_instant[i] = u[i, lengths[i].item()-1]

    u = torch.nn.utils.rnn.pack_padded_sequence(u,
                                                lengths,
                                                batch_first=True,
                                                enforce_sorted=True)

    x0 = x0.to(device)
    y = y.to(device)
    t = t.to(device)
    u = u.to(device)
    deltas = deltas.to(device)
    u_instant = u_instant.to(device)
    deltasPhys = deltasPhys.to(device)
    deltasPhys.requires_grad = True  
    delta_window = delta_window.to(device)
    # deltasPhys.requires_grad = True  

    return x0, y, u, deltas, deltasPhys, u_instant, t, delta_window


def DON_prep_inputs(par, t_batch, u_batch, r0_batch, r_batch, device):
    t_batch = t_batch.to(device) 
    r_batch = r_batch.to(device)
    u_batch = u_batch.to(device)
    r0_batch = r0_batch.to(device)
            
    if par.PI_ML: # WITH PHYS
        t_phys = torch.rand_like(t_batch)*par.u_delta
        t_phys = t_phys.to(device)
    else:
        t_phys = t_batch	
    t_phys.requires_grad = True    

    return t_batch, r_batch, u_batch, r0_batch, t_phys


def log_progress(logging, epoch, train_loss_NN, train_loss_PHYS, val_loss_NN, val_loss_PHYS, gain, phys_param, par):

    logging['loss_train_NN'].append(train_loss_NN)
    logging['loss_train_PHYS'].append(train_loss_PHYS)
    logging['loss_train_tot'].append(train_loss_NN+gain.item()*train_loss_PHYS)
    logging['gain'].append(gain.item())
    
    param_epoch = []
    for i, p in enumerate(phys_param):
        if i==3:
            param_epoch.append(np.asarray(p.data.clone().tolist())) 
        else:
            param_epoch.append(np.exp(p.data.clone().tolist())) 
                
    logging['Phys_par'].append(param_epoch)

    logging['loss_val_NN'].append(val_loss_NN)
    logging['loss_val_PHYS'].append(val_loss_PHYS)
    logging['loss_val_tot'].append(val_loss_NN+gain.item()*val_loss_PHYS)

    if epoch % par.log_iterations == 0:
        print(f"\n Epoch {epoch}: \n Train Loss data: {train_loss_NN:.4f}, Val Loss data: {val_loss_NN:.4f}")
        print(f"Train Loss physics: {train_loss_PHYS:.4f}, Val Loss physics: {val_loss_PHYS:.4f}, \n gain: {gain.item()} \n ------- \n")

    return logging #return the updated logging
