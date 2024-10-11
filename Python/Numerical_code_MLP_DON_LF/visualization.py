import os
import matplotlib.pyplot as plt
import numpy as np

#########################################################################

def plot_training(logging, par):
    fig1, axs1 = plt.subplots(1,2, figsize=(15,6))
    
    train_epochs = len(logging['loss_train_NN'])

    axs1[0].plot(np.arange(train_epochs)+1, logging['loss_train_NN'], label='Train data loss', color=f'C{0}')
    axs1[0].plot(np.arange(train_epochs)+1, logging['loss_train_PHYS'], label='Train physics loss', color= f'C{2}', alpha = 0.75) #f'#629FCA') #78ADD2
    axs1[0].plot(np.arange(train_epochs)+1, logging['loss_val_NN'], label='Val data loss', color=f'C{1}') 
    axs1[0].plot(np.arange(train_epochs)+1, logging['loss_val_PHYS'], label='Val physics loss', color=f'C{3}', alpha = 0.75) # f'#FFA556') #FFB26E

    axs1[0].set_ylabel('Loss [-]')
    axs1[0].set_title('Losses during training')
    axs1[0].legend()
    axs1[0].grid(True)
    axs1[0].set_yscale('log')
    axs1[0].set_xlabel('Epoch')
    axs1[0].set_xlim(1, train_epochs)

    axs1[1].plot(np.arange(train_epochs)+1, logging['gain'], label='Gain', color=f'C{4}')
    axs1[1].set_ylabel('$\lambda$ [-]')
    axs1[1].set_xlabel('Epoch')
    axs1[1].set_title('Physics gain')
    axs1[1].set_xlim(1, train_epochs)
    axs1[1].grid(True)

    return fig1

#########################################################################

def plot_results(test_results, par):

    t_array = np.arange(par.t0, par.tend + par.st, par.st)

    fig2, axs2 = plt.subplots(par.N_r,1, figsize=(10,par.N_r*2))
    for i in range(par.N_r):
        axs2[i].plot(t_array, test_results['r_batch'][:,i], label='Data', color=f'C{0}')
        axs2[i].plot(t_array, test_results['r_est'][:,i], label='Estimation', color=f'C{1}')
        axs2[i].plot(t_array, test_results['u_batch'][:,i], label='Stimulation', color=f'C{2}')
        axs2[i].set_ylabel('r/u [Hz]/[mA]')
        axs2[i].set_title(f'Electrode {i+1}')
        axs2[i].grid(True)
    axs2[i].legend()
    axs2[i].set_xlabel('Time [ms]')

    return fig2

#########################################################################