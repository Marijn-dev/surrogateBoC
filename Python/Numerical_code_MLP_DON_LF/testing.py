import os
import torch
from tqdm import tqdm
from train import lf_prep_inputs, DON_prep_inputs
import numpy as np

def test_model(modelNN, modelPhys, test_loader, criterion, par, overwrite_arch = ''):
    #optionally overwrite the setting of configuration
    if overwrite_arch == '':
        arch = par.architect
    else:
        arch = overwrite_arch
    
    if arch == 'DON' and par.u_delta_fixed:
        return DON_test_model(modelNN, modelPhys, test_loader, criterion, par)
    elif arch == 'DON_LH':
        return DON_LH_test_model(modelNN, modelPhys, test_loader, criterion, par)
    elif arch == 'LF' and par.u_delta_fixed:
        return LF_test_model(modelNN, modelPhys, test_loader, criterion, par)
    else:
        raise ValueError('The architecture is not implemented yet.')
    

def test_single_sample(modelNN, modelPhys, criterion, sample, par):
    if par.architect == 'DON' and par.u_delta_fixed:
        return calc_loss_single_sample_DON(modelNN, modelPhys, criterion, sample, par)
    elif par.architect == 'LF' and par.u_delta_fixed:
        return calc_loss_single_sample_LF(modelNN, modelPhys, criterion, sample, par)


##################################################################################
# Test function for different model architectures
##################################################################################

def DON_test_model(modelNN, modelPhys, test_loader, criterion, par):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelNN.eval()
    modelPhys.eval()

    running_loss_data = 0.0
    running_loss_phys = 0.0
    
    N_horizon = int(par.u_delta/par.st) 

    for _, (sample) in tqdm(enumerate(test_loader)):
        t_batch, r_batch, u_batch, r0_batch, t_phys = DON_prep_inputs(par, *sample, device)
        
        r_est = torch.zeros_like(r_batch)
        r_phys = torch.zeros_like(r_batch)

        r_init = r0_batch[0]
        for i in range(t_batch.shape[0]):
            r_est[i] = modelNN(t_batch[i], u_batch[i], r_init)
            
            if par.PI_ML: # WITH PHYS
                r_phys[i] = modelNN(t_phys[i], u_batch[i], r_init)
            
            #Update the initial value
            if (i+1) % N_horizon == 0:
                r_end_horizon = modelNN(par.u_delta*torch.ones_like(t_batch[i]), u_batch[i], r_init)
                r_init = r_end_horizon

        if par.PI_ML: # WITH PHYS
            output_phys = modelPhys(r_phys, u_batch)
            grads = []
            for g in range(par.N_r):
                grad = torch.autograd.grad(r_phys[:, g], t_phys, torch.ones_like(r_phys[:, g]), create_graph=True)[0]
                grads.append(grad)
            dr_phys_dt = torch.cat(grads, dim=1)
        else: # WITHOUT PHYS
            output_phys = torch.zeros_like(r_batch).to(device)
            dr_phys_dt = torch.zeros_like(r_batch).to(device)

        loss_data = criterion(r_est, r_batch)
        loss_phys = criterion(dr_phys_dt-output_phys,torch.zeros_like(dr_phys_dt))

        running_loss_data += loss_data.item()
        running_loss_phys += loss_phys.item()

    test_loss_data = running_loss_data/len(test_loader)
    test_loss_phys = running_loss_phys/len(test_loader)

    t_array = np.arange(par.t0, par.tend + par.st, par.st)

    #store last example
    logging = {"loss_data": test_loss_data, "loss_phys": test_loss_phys,
               "t_batch": t_array, "u_batch": u_batch.cpu().detach().numpy(), "r0_batch": r0_batch.cpu().detach().numpy(), "r_batch": r_batch.cpu().detach().numpy(), 
               "r_est": r_est.cpu().detach().numpy(), "r_phys": r_phys.cpu().detach().numpy(), "t_phys": t_phys.cpu().detach().numpy(), 
               "output_phys": output_phys.cpu().detach().numpy(), "dr_phys_dt": dr_phys_dt.cpu().detach().numpy()}

    #print the test losses
    print(f"Test Loss data: {test_loss_data:.4f}, Test Loss physics: {test_loss_phys:.4f}")

    return logging

###########################################################################
def DON_LH_test_model(modelNN, modelPhys, test_loader, criterion, par):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelNN.eval()
    modelPhys.eval()

    running_loss_data = 0.0
    running_loss_phys = 0.0

    for _, (t_batch, u_batch, r0_batch, r_batch) in tqdm(enumerate(test_loader)):
        t_batch = t_batch.to(device) 
        u_batch = u_batch.to(device)
        r0_batch = r0_batch.to(device) 
        r_batch = r_batch.to(device)
        
        r_est = torch.zeros_like(r_batch)
        r_phys = torch.zeros_like(r_batch)
        t_phys = torch.rand_like(t_batch)*par.u_delta
        t_phys = t_phys.to(device)
        t_phys.requires_grad = True

        r_init = r0_batch[0]
        u_prev = u_batch[0]

        for i in range(t_batch.shape[0]):            
            if any(abs(u_prev - u_batch[i])>1e-4):
                #Update the initial value
                t_end_horizon = t_batch[i-1] + par.st
                r_end_horizon = modelNN(t_end_horizon, u_batch[i-1], r_init)
                r_init = r_end_horizon
                u_prev = u_batch[i]
            
            r_est[i] = modelNN(t_batch[i], u_batch[i], r_init)
            r_phys[i] = modelNN(t_phys[i], u_batch[i], r_init)

        output_phys = modelPhys(r_phys, u_batch, t_phys)
        grads = []
        for g in range(par.N_r):
            grad = torch.autograd.grad(r_phys[:, g], t_phys, torch.ones_like(r_phys[:, g]), create_graph=True)[0]
            grads.append(grad)
        dr_phys_dt = torch.cat(grads, dim=1)

        loss_data = criterion(r_est, r_batch)
        loss_phys = criterion(dr_phys_dt-output_phys,torch.zeros_like(dr_phys_dt))

        running_loss_data += loss_data.item()
        running_loss_phys += loss_phys.item()

    test_loss_data = running_loss_data/len(test_loader)
    test_loss_phys = running_loss_phys/len(test_loader)

    t_array = np.arange(par.t0, par.tend + par.st, par.st)

    #store last example
    logging = {"loss_data": test_loss_data, "loss_phys": test_loss_phys,
               "t_batch": t_array, "u_batch": u_batch.cpu().detach().numpy(), "r0_batch": r0_batch.cpu().detach().numpy(), "r_batch": r_batch.cpu().detach().numpy(), 
               "r_est": r_est.cpu().detach().numpy(), "r_phys": r_phys.cpu().detach().numpy(), "t_phys": t_phys.cpu().detach().numpy(), 
               "output_phys": output_phys.cpu().detach().numpy(), "dr_phys_dt": dr_phys_dt.cpu().detach().numpy()}

    #print the test losses
    print(f"Test Loss data: {test_loss_data:.4f}, Test Loss physics: {test_loss_phys:.4f}")

    return logging
    
###########################################################################

def LF_test_model(modelNN, modelPhys, test_loader, criterion, par):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelNN.eval()
    modelPhys.eval()

    running_loss_data = 0.0
    running_loss_phys = 0.0

    for _, (sample) in tqdm(enumerate(test_loader)):
        r0, r_batch, u_inn, deltas, deltasPhys, u_inst, t_batch = lf_prep_inputs(*sample, device, par)
        
        # deltas.requires_grad = True

        r_est = modelNN(r0, u_inn, deltas)
        loss_data = criterion(r_est, r_batch)

        if par.PI_ML: # WITH PHYS
            # Physics Loss 
            r_phys = modelNN(r0, u_inn, deltasPhys)
            output_phys = modelPhys(r_phys, u_inst)

            # output_phys = modelPhys(r_est, u_inst)

            grads = []
            for i in range(par.N_r):
                gradient_r = torch.autograd.grad(r_phys[:, i], deltasPhys, torch.ones_like(r_phys[:, i]), create_graph=True)[0]
                gradient_r = torch.sum(gradient_r, (-2,-1))*(1/par.u_delta)
                grads.append(gradient_r)
            dr_phys_dt = torch.stack(grads, dim=1)

            loss_phys = criterion(dr_phys_dt-output_phys,torch.zeros_like(dr_phys_dt))
        
        else: # WITHOUT PHYS
            output_phys = torch.zeros_like(r_batch).to(device)
            dr_phys_dt = torch.zeros_like(r_batch).to(device)
            loss_phys = torch.tensor(0.0).to(device)
        
        running_loss_data += loss_data.item()
        running_loss_phys += loss_phys.item()

    test_loss_data = running_loss_data/len(test_loader)
    test_loss_phys = running_loss_phys/len(test_loader)
    
    #sort the test sample data
    sort_idxs_t = torch.argsort(t_batch[:,0])
    t_batch = t_batch[sort_idxs_t]
    r_batch = r_batch[sort_idxs_t]
    u_inst = u_inst[sort_idxs_t]
    r_est = r_est[sort_idxs_t]
    output_phys = output_phys[sort_idxs_t]
    dr_phys_dt = dr_phys_dt[sort_idxs_t]

    #store last example
    logging = {"loss_data": test_loss_data, "loss_phys": test_loss_phys,
               "t_batch": t_batch.cpu().detach().numpy(), "u_batch": u_inst.cpu().detach().numpy(), "r_batch": r_batch.cpu().detach().numpy(), 
               "r_est": r_est.cpu().detach().numpy(), "output_phys": output_phys.cpu().detach().numpy(), "dr_dt": dr_phys_dt.cpu().detach().numpy()}

    #print the test losses
    print(f"Test Loss data: {test_loss_data:.4f}, Test Loss physics: {test_loss_phys:.4f}")

    return logging



###########################################################################
#create a function to calculate the loss for a single sample

#DON version
def calc_loss_single_sample_DON(modelNN, modelPhys, criterion, sample, par):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modelNN.eval()
    modelPhys.eval()
    
    N_horizon = int(par.u_delta/par.st) 
    
    t_batch, r_batch, u_batch, r0_batch, t_phys = DON_prep_inputs(par, *sample, device)

    r_est = torch.zeros_like(r_batch)
    r_phys = torch.zeros_like(r_batch)
    
    r_init = r0_batch[0]
    for i in range(t_batch.shape[0]):
        r_est[i] = modelNN(t_batch[i], u_batch[i], r_init)
        r_phys[i] = modelNN(t_phys[i], u_batch[i], r_init)
        
        #Update the initial value
        if (i+1) % N_horizon == 0:
            r_end_horizon = modelNN(par.u_delta*torch.ones_like(t_batch[i]), u_batch[i], r_init)
            r_init = r_end_horizon

    output_phys = modelPhys(r_phys, u_batch)
    grads = []
    for g in range(par.N_r):
        grad = torch.autograd.grad(r_phys[:, g], t_phys, torch.ones_like(r_phys[:, g]), create_graph=True)[0]
        grads.append(grad)
    dr_phys_dt = torch.cat(grads, dim=1)

    loss_data = criterion(r_est, r_batch)
    loss_phys = criterion(dr_phys_dt-output_phys,torch.zeros_like(dr_phys_dt))

    t_array = np.arange(par.t0, par.tend + par.st, par.st)

    traj_log = {'r_est': r_est.detach().cpu().numpy(), 'r': r_batch.detach().cpu().numpy(), 
                't': t_array, 'u': u_batch.detach().cpu().numpy()}

    return loss_data.item(), loss_phys.item(), traj_log


# Learning flow version
def calc_loss_single_sample_LF(modelNN, modelPhys, criterion, sample, par):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    modelNN.eval()
    modelPhys.eval()

    r0, r_batch, u_inn, deltas, deltasPhys, u_inst, t_batch = lf_prep_inputs(*sample, device, par)

    r_est = modelNN(r0, u_inn, deltas)
    loss_data = criterion(r_est, r_batch)

    if par.PI_ML: # WITH PHYS
        # Physics Loss 
        r_phys = modelNN(r0, u_inn, deltasPhys)
        output_phys = modelPhys(r_phys, u_inst)

        grads = []
        for i in range(par.N_r):
            gradient_r = torch.autograd.grad(r_phys[:, i], deltasPhys, torch.ones_like(r_phys[:, i]), create_graph=True)[0]
            gradient_r = torch.sum(gradient_r, (-2,-1))*(1/par.u_delta)
            grads.append(gradient_r)
        dr_phys_dt = torch.stack(grads, dim=1)

        loss_phys = criterion(dr_phys_dt-output_phys,torch.zeros_like(dr_phys_dt))
    
    else: # WITHOUT PHYS
        output_phys = torch.zeros_like(r_batch).to(device)
        dr_phys_dt = torch.zeros_like(r_batch).to(device)
        loss_phys = torch.tensor(0.0).to(device)

    sort_idxs_t = torch.argsort(t_batch[:,0])
    t_batch = t_batch[sort_idxs_t]
    r_batch = r_batch[sort_idxs_t]
    u_inst = u_inst[sort_idxs_t]
    r_est = r_est[sort_idxs_t]

    traj_log = {'r_est': r_est.detach().cpu().numpy(), 'r': r_batch.detach().cpu().numpy(), 
                't': t_batch[:,0].detach().cpu().numpy(), 'u': u_inst.detach().cpu().numpy()}

    return loss_data.item(), loss_phys.item(), traj_log