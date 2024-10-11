import os
import scipy.io as sio
import sys

from config import Config
from data_generation import generate_data
from data_preparation import prepare_data, save_dataloaders
from model import *

def main():
    # Initialize all settings
    config = Config()

    # Generate data
    t, r, u, u_props = generate_data(config, config.N_data_seq)
    
    # Prepare and store dataloaders for DON
    config.architect = 'DON'
    train_loader, val_loader, test_loader = prepare_data(config, t, r, u, u_props)
    save_dataloaders(config, train_loader, val_loader, test_loader)

    # Prepare and store dataloaders for LF    
    config.architect = 'LF'
    train_loader, val_loader, test_loader = prepare_data(config, t, r, u, u_props)
    save_dataloaders(config, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()