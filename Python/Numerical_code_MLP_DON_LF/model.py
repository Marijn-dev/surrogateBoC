import torch.nn as nn
import torch.optim as optim
import torch

##################################################################################
# Model Initialization
##################################################################################

def initialize_model(par, overwrite_arch = ''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computations done using {device}")

    #optionally overwrite the setting of configuration
    if overwrite_arch == '':
        arch = par.architect
    else:
        arch = overwrite_arch
    
    # Initalise data and physics model
    if arch == 'MLP':
        modelNN = MLP(1, par.N_r, par.n_hidden, par.n_layers)
    elif arch == 'DON' or arch == 'DON_LH':
        modelNN = DON_model(par, par.n_out_branch_trunk)
    elif arch == 'LF' and par.u_delta_fixed:
        modelNN = LF_model(par)
    else:
        raise ValueError('The architecture is not implemented yet.')
    modelPhys = PHYS_model(par)
    
    #Send to tdevice
    modelNN.to(device)
    modelPhys.to(device)
    gain =  nn.Parameter(torch.Tensor(1).fill_(par.init_gain).to(device))

    # Initialise optimizers
    if par.PI_ML:
        optimizer = torch.optim.Adam(
            [
                {'params': modelNN.parameters()},
                {'params': modelPhys.parameters(), 'lr': par.eta_phys_par} #par.eta_phys_par
            ],
            lr=par.eta,  # base learning rate for NN1 parameters
            weight_decay=par.weight_decay
        )    
    else:
        optimizer = optim.Adam(modelNN.parameters(), lr=par.eta, weight_decay=par.weight_decay)
    
    optimizer_gain = optim.Adam([gain], lr=par.eta_gain, maximize=True)

    # Initialise the loss function
    criterion = nn.MSELoss()

    # Setup the learning rate scheduler 
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    mode = 'min',  
    factor = 1/par.shed_factor, 
    cooldown = par.shed_cooldown,  
    patience = par.shed_patience,  
    threshold = par.es_delta)  

    # Intialise the early stopping
    early_stop = EarlyStopping(par.es_patience, par.es_delta, par.PI_ML)

    return modelNN, modelPhys, gain, criterion, optimizer, optimizer_gain, sched, early_stop


##################################################################################
# Different Model Architects
##################################################################################

### Simple MLP model
class MLP(nn.Module):
    "Model used for the resemblence of the data"
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(n_input, n_hidden),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(n_hidden, n_hidden),
                            activation()]) for _ in range(n_layers-1)])
        self.fce = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

### Deep Operator Network model
class DON_model(nn.Module):
    def __init__(self, par, n_output=60):
        super(DON_model, self).__init__()
        self.par = par
        if par.architect == 'DON':
            self.branch_1 = MLP(par.N_r, n_output, par.n_hidden, par.n_layers) #the u() input
        elif par.architect == 'DON_LH':
            self.branch_1 = MLP(par.N_r+1, n_output, par.n_hidden, par.n_layers) #the u() input
        self.branch_2 = MLP(par.N_r, n_output, par.n_hidden, par.n_layers) #the r_0 input
        self.trunk = MLP(1, n_output, par.n_hidden, par.n_layers)

    def forward(self, x1, x2, x3):
        # x2 = self.branch_1(x2[...,0:self.par.N_u]) #the u() input only nonzero part
        x2 = self.branch_1(x2) #the u() input also nonzero part
        x3 = self.branch_2(x3) #the r_0 input
        x_branch = x2*x3
        x_trunk = self.trunk(x1)

        num_splits = x_branch.size(-1) // self.par.N_r
        B_splits = torch.split(x_branch, num_splits, dim=-1)
        T_splits = torch.split(x_trunk, num_splits, dim=-1)

        r_list = []
        for i in range(self.par.N_r):
            r_list.append(torch.sum(B_splits[i] * T_splits[i], dim=-1))
        x = torch.stack(r_list, dim=-1)

        return x

### Learning flow function model
class LF_model(nn.Module):
    
    def __init__(self, par):
        super(LF_model, self).__init__()

        self.control_rnn_size = par.control_rnn_size
        self.control_rnn_depth = par.control_rnn_depth

        self.u_rnn = torch.nn.LSTM(
            input_size= par.N_r + 1,  # or 2 * par.N_r
            hidden_size= self.control_rnn_size,
            batch_first=True,
            num_layers=self.control_rnn_depth,
            dropout=0, 
        )

        x_dnn_osz = self.control_rnn_depth * self.control_rnn_size
        u_dnn_isz = self.control_rnn_size

        self.x_dnn = MLP(par.N_r, x_dnn_osz, par.enc_dec_hidden, par.enc_dec_depth)
        self.u_dnn = MLP(u_dnn_isz, par.N_r, par.enc_dec_hidden, par.enc_dec_depth)

    def forward(self, x, rnn_input, deltas):
        h0 = self.x_dnn(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed,
                                                           batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h
        output = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]),
                                             h_lens - 1, :])
        return output


### Physics model
class PHYS_model(nn.Module):
    def __init__(self, par):
        super(PHYS_model, self).__init__()
        self.par = par

        if self.par.PI_ML_learn_par:
            self.alpha = nn.Parameter(torch.log(torch.ones(self.par.N_r)), requires_grad=True) 
            self.beta = nn.Parameter(torch.log(torch.rand(self.par.N_r)*10), requires_grad=True) 
            self.gamma = nn.Parameter(torch.log(torch.rand(self.par.N_r)), requires_grad=True) 
            self.W = nn.Parameter(torch.rand(self.par.N_r*(self.par.N_r-1)), requires_grad=True)

        else:
            self.alpha = torch.tensor(self.par.alpha).float()
            self.beta = torch.tensor(self.par.beta).float()
            self.gamma = torch.tensor(self.par.gamma).float()
            self.W = torch.tensor(self.par.W_flatten).float()

        if self.par.phys_dist != 0:
            W_rand = torch.randn_like(self.W)
            current_norm = torch.norm(W_rand,p='fro') 
            self.W_dist = W_rand*(self.par.phys_dist/current_norm)
        else:
            self.W_dist = torch.zeros_like(self.W)

    def forward(self, r, u, t = None):  

        if self.par.PI_ML_learn_par: #trick of training the log of the parameters
            alpha_eq = torch.exp(self.alpha)
            beta_eq = torch.exp(self.beta)
            gamma_eq = torch.exp(self.gamma)
        else:
            alpha_eq = self.alpha
            beta_eq = self.beta
            gamma_eq = self.gamma

        I = torch.zeros_like(r)
        index = 0
        for i in range(self.par.N_r):
            for j in range(self.par.N_r):
                if i!=j:
                    I[:,i] += (self.W[index]+self.W_dist[index])*r[:,j].t()
                    index += 1

        if self.par.architect == 'DON_LH':
            for l in range(t.shape[0]):
                if (t[l]<=u[l,-1]):
                    I[l] += u[l,0:self.par.N_r]
        else:
            I += u
            
        drdt = torch.zeros_like(r)
        for i in range(self.par.N_r):
            drdt[:,i] = - alpha_eq[i]*r[:,i] + beta_eq[i]/(1+torch.exp(-I[:,i] + gamma_eq[i]))

        return drdt

##################################################################################
# Helper functions
##################################################################################

#print the model structure
def print_model_structure(model):
    total_trainable_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print("Model's trainable parameters:")
    for param_tensor in model.parameters():
        print(param_tensor.size())
    print(f"Total trainable parameters: {total_trainable_par}")


class EarlyStopping:
    def __init__(self, es_patience, es_delta, PI_ML_bool):
        self.patience = es_patience
        self.delta = es_delta
        self.PI_ML_bool = PI_ML_bool

        self.best_lossNN = float('inf')
        self.best_lossPhys = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model = False

    def step(self, val_lossNN, val_lossPhys):
        self.best_model = False

        if (self.best_lossNN - val_lossNN > self.delta) or ((self.best_lossPhys - val_lossPhys > self.delta) and self.PI_ML_bool):
            self.best_lossNN = val_lossNN
            self.best_lossPhys = val_lossPhys
            self.best_model = True
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True