from scipy.integrate import solve_ivp
import numpy as np
import torch 

def generate_data(par, N=1):
    """Generate data for the model.

    Outputs:
    --------
    u_samples: input firing rate samples
    r_samples: output firing rate samples
    """
    t_array = np.arange(par.t0, par.tend + par.st, par.st)
    r0 = np.random.uniform(0, 10, par.N_r)     #r0 = np.zeros(par.N_r) 

    #Solve the ODE for different input stimulation patterns
    sols_r = []
    sols_u = []
    sols_u_seq = []

    for _ in range(N):
        if par.u_type == 'pulse':
            u_seq, u = generate_sparse_input(par)
        elif par.u_type == 'sinusoidal':
            u_seq, u = generate_sinusoidal_input(par)

        sol = solve_ivp(lambda t, r: _FR_dynamics_ODE(t, r, par, u_seq), t_span=[par.t0, t_array[-1]], y0=r0, t_eval=t_array, method='RK45', max_step=par.u_delta/10)
        
        sols_r.append(sol.y)
        sols_u.append(u)
        sols_u_seq.append(u_seq)    

    #Reorganise the data in torch tensors
    t_t, t_r, t_u, t_u_seq = _convert_to_tensor(t_array, sols_r, sols_u, sols_u_seq)

    return t_t, t_r, t_u, t_u_seq 
  
  
############################################################
    # Helper functions
############################################################

def _FR_dynamics_ODE(t, r, par, u_seq):
    """Compute the dynamics of the ODE system.
    Args:
    ----
    t: time
    y: state
    u_t: input properties of this sequence
    
    Outputs:
    --------
    dydt: derivative of the state
    """

    #Get the stimulation input
    u_vec = get_piecewise_value(par, t, u_seq)

    #Multiple unit dynamics
    I = par.W @ r + u_vec
    
    drdt = np.zeros_like(r)

    for i in range(par.N_r):
        drdt[i] = -par.alpha[i]*r[i] + par.beta[i]/(1+np.exp(-I[i] + par.gamma[i]))
    return drdt


def generate_sparse_input(par):
    """
    Generate a sparse input array where each element is zero with probability (1 - probability),
    or sampled from a uniform distribution [low, high] with probability.
    
    :param size: Number of time instances
    :param probability: Probability of a non-zero value
    :param low: Lower bound of the uniform distribution
    :param high: Upper bound of the uniform distribution
    :return: Array of generated values
    """

    size = int(np.ceil((par.tend - par.t0 + par.st) / par.u_delta))

    # Generate random uniform values
    uniform_values = np.random.uniform(par.u_amp_min, par.u_amp_max, (size,par.N_r))
    # uniform_values = np.random.lognormal(np.log(15), 0.5, (size,par.N_r))

    # Generate a mask for non-zero values
    mask = np.random.rand(size,par.N_r) < par.u_density

    if par.N_u != par.N_r:
        mask_out = par.N_r - par.N_u
        mask[:,-mask_out:] = 0

    # Apply mask to the uniform values
    u_seq = np.where(mask, uniform_values, 0)
    
    # u = np.repeat(u_seq, times)
    t = np.arange(par.t0, par.tend + par.st, par.st)
    u = np.zeros((len(t), par.N_r))

    for i in range(len(t)):
        u[i] = get_piecewise_value(par, t[i], u_seq)

    return u_seq, u


def generate_sinusoidal_input(par):
    """
    Generate a sinusoidal input array based on the specified parameters, with
    A following a log-normal distribution and Omega following a uniform distribution.
    
    :param par: A parameter object with attributes:
    :return: Tuple of (u_seq, u), where
             u_seq is the generated sinusoidal sequence, and
             u is the repeated sequence over time steps.
    """
    
    size = int(np.ceil((par.tend - par.t0 + par.st) / par.u_delta))
    
    u_seq = np.zeros((size, par.N_r))

    for i in range(par.N_u):
        A = np.random.lognormal(mean=np.log(10), sigma=0.5)
        # A = np.random.uniform(par.u_amp_min, par.u_amp_max)
        Omega = np.random.uniform(0, 2 * np.pi / 10)
        
        # Generate the time index array
        k = np.arange(size)
        
        # Generate the sinusoidal sequence
        u_seq[:,i] = A * np.sin(Omega * k)
        u_seq[:,i] = np.maximum(u_seq[:,i], 0)
    
    t = np.arange(par.t0, par.tend + par.st, par.st)
    u = np.zeros((len(t), par.N_r))

    for i in range(len(t)):
        u[i] = get_piecewise_value(par, t[i], u_seq)

    return u_seq, u


def get_piecewise_value(par, time_instance, u_seq):
    """
    Retrieve the value from a piecewise constant array given a specific time instance.

    :param time_instance: The specific time instance to query
    :param start_times: An array with the start times of each constant segment
    :param values: An array with the corresponding values for each constant segment
    :return: The value at the given time instance
    """

    # Array with the start time of each of the control segments
    t_u = np.arange(par.t0, par.tend+par.st, par.u_delta)

    # Find the index of the last start time that is less than or equal to the given time instance
    if time_instance < t_u[0]:
        index = 0
    else:
        index = np.searchsorted(t_u, time_instance, side='right') - 1
    
    return u_seq[index]



def _convert_to_tensor(t_array, r,u,u_props):
    """Convert the data to torch tensors.

    Args:
    ----
    r: output firing rate samples
    u: input firing rate samples
    u_props: input firing rate properties

    Outputs:
    --------
    t_r: output firing rate samples
    t_u: input firing rate samples
    t_u_props: input firing rate properties
    """
    
    t_r = torch.stack([torch.from_numpy(result) for result in r])
    t_u = torch.stack([torch.from_numpy(result) for result in u])
    t_u_props = torch.stack([torch.from_numpy(result) for result in u_props])
    t_r = t_r.permute(0, 2, 1)
    t_t = torch.stack([torch.from_numpy(t_array) for _ in range(len(r))])
    t_t.unsqueeze_(2)

    t_r = t_r.type(torch.FloatTensor)
    t_u = t_u.type(torch.FloatTensor)
    t_u_props = t_u_props.type(torch.FloatTensor)
    t_t = t_t.type(torch.FloatTensor)

    return t_t, t_r, t_u, t_u_props