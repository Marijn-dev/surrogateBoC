import numpy as np
import wandb

class Config:
    def __init__(self):     
        
        # General parameters
        self.save_results = True 
        self.exp_name = "scale_up_LF"
        self.sim_number = 23
        self.architect = 'LF' # Choose model architecture: 'MLP' / 'DON' / 'LF' / 'DON_LH'
      
        # Parameters data generation
        self.N_r = 3
        self.N_u = 1
        self.N_data_seq = 50 #minimum of 5 sequences to be able to make the split with test/val/train

        self.t0 = 0 #[s]
        self.tend = 20 #[s]
        self.st = 1/10 #[s]
    
        self.noise = False
        self.noise_mu = 0
        self.noise_sigma = 0.5

        self.u_amp_min = 10 #[mA]
        self.u_amp_max = 20 #[mA]
        self.u_delta_fixed = True
        self.u_delta = 0.4 #[ms] Should be multiple or equal to st! 2*self.st
        self.u_delta_min = 0.3 #[mA] Should be multiple or equal to st!
        self.u_delta_max = 0.5 #[mA] Should be multiple or equal to st! 
        self.u_pause_next = 0.2 #[ms] Should be multiple or equal to st!
        self.u_density = 0.2 #the density of the amount of stimulations. 1: always stimulation, 0: never stimulation
        self.u_type = 'pulse' # 'pulse' or 'sinusoidal'
    
        # General Parameters ML
        self.PI_ML = True  # True: use the physics informed loss and data loss, False: use only the data loss
        self.PI_ML_learn_par = True # True: learn the physics parameters, False: use the preset physics parameters
        self.num_epochs = 1000
        self.log_iterations = 50
        self.eta = 0.01
        self.eta_phys_par = 0.01
        self.eta_gain = 0.005
        self.weight_decay = 1e-4
        self.test_split_ratio = 0.2
        self.val_split_ratio = 0.2
        self.bs_train = 512 
        self.bs_val = 512 
        self.bs_test = int(np.ceil((self.tend-self.t0+self.st)/self.st))
        self.init_gain = 0.1 #initial gain value
        self.start_gain_update = 1 #number of epochs after which the gain will be updated
        self.shed_factor = 2 #factor by which the learning rate will be reduced
        self.shed_cooldown = 10 #number of epochs to wait before resuming normal operation after lr has been reduced
        self.shed_patience = 15 #number of epochs with no improvement after which learning rate will be reduced
        self.es_delta = 0.0001 #minimum change to consider as an improvement
        self.es_patience = 150 #number of epochs with no improvement after which training will be stopped
        self.phys_dist = 0 # The std of the normal distribution used to disturb the W in the physics parameters
        
        # ML parameters if DeepONet is used
        self.n_hidden = 60 
        self.n_out_branch_trunk = 60 #should be a multiple of N_r
        self.n_layers = 4
        
        # ML parameters if Learning Flow function is used
        self.control_rnn_size = 24
        self.control_rnn_depth = 1
        self.enc_dec_depth = 2
        self.enc_dec_hidden = self.control_rnn_size

        # General parameters
        self.save_name = f"{self.sim_number}_{self.architect}_PI{self.PI_ML}_Nr{self.N_r}_Nu{self.N_u}_Nseq{self.N_data_seq}"

        self.save_folder = f"/home/marijn/Documents/Thesis/Results_simulations/{self.exp_name}/Sim_{self.sim_number}"
        self.folder_datasets = f"/home/marijn/Documents/Thesis/Datasets"
        
 
        #self.save_folder = f"/Results_simulations/{self.exp_name}/Sim_{self.sim_number}"
        #self.folder_datasets = f"/Datasets"

        # Physics parameters
        self.preset_phys_values = True
        if self.preset_phys_values and self.N_r == 3:
            self.N_r = 3
            self.W= np.array([[0, 0, 0], [0.8, 0, 0], [1.5, 0, 0]])
            self.W_flatten= np.array([0, 0, 0.8, 0, 1.5, 0])
            self.alpha = np.array([1.8, 0.8, 2.5])
            self.beta = np.array([25, 30, 20])
            self.gamma = np.array([7, 5, 5])
        else:
            self.W = self._connected_to_stim(self.N_r, 1,0.1)
            self.W_flatten = self._flatten_excluding_diagonal(self.W)
            self.alpha = self._generate_random_par_array(self.N_r,0.6,3)
            self.beta = self._generate_random_par_array(self.N_r,15,35)
            self.gamma = self._generate_random_par_array(self.N_r,5,9)


    ########################################################
    # Functions for handling the intialisation of the model
    ########################################################

    def update_phys_par(self):        
        if self.preset_phys_values and self.N_r == 3:
            self.N_r = 3
            self.W= np.array([[0, 0, 0], [0.8, 0, 0], [1.5, 0, 0]])
            self.W_flatten= np.array([0, 0, 0.8, 0, 1.5, 0])
            self.alpha = np.array([1.8, 0.8, 2.5])
            self.beta = np.array([25, 30, 20])
            self.gamma = np.array([7, 5, 5])
        else:
            self.W = self._connected_to_stim(self.N_r, 1,0.1)
            self.W_flatten = self._flatten_excluding_diagonal(self.W)
            self.alpha = self._generate_random_par_array(self.N_r,0.6,3)
            self.beta = self._generate_random_par_array(self.N_r,15,35)
            self.gamma = self._generate_random_par_array(self.N_r,5,9)

    # Helper functions for the physics parameters
    def _connected_to_stim(self,N_r,mu,std):
        matrix = np.zeros((N_r, N_r))
        for i in range(N_r):
            for j in range(1):
                matrix[i, j] = np.random.normal(mu,std)
        matrix[0,0] = 0
        return matrix
    
    def _flatten_excluding_diagonal(self, matrix):
        N_r = matrix.shape[0]
        flattened = []
        for i in range(N_r):
            for j in range(N_r):
                if i != j:
                    flattened.append(matrix[i, j])
        return np.array(flattened)

    def _generate_random_par_array(self, N_r, low, high):
        param = np.random.uniform(low,high,N_r)
        return param
         
    #Helper function to print the physics parameters
    def print_physics_parameters(self):
        print(f"alpha: {self.alpha}")
        print(f"beta: {self.beta}")
        print(f"gamma: {self.gamma}")
        print(f"W: {self.W}")

    # Save the initialisation settings
    def save_initialisation_settings(self):      
        print('Saving the initialisation settings')  
        
        with open(f"{self.save_folder}/{self.save_name}_Initialisation.txt", "w") as f:
            f.write('General parameters\n')
            f.write(f"save_results: {self.save_results}\n")
            f.write(f"sim_number: {self.sim_number}\n")
            f.write(f"architect: {self.architect}\n\n")

            f.write('Data generation parameters\n')
            f.write(f"N_r: {self.N_r}\n")
            f.write(f"N_u: {self.N_u}\n")
            f.write(f"N_data_seq: {self.N_data_seq}\n")
            f.write(f"t0: {self.t0}\n")
            f.write(f"tend: {self.tend}\n")
            f.write(f"st: {self.st}\n")
            f.write(f"noise: {self.noise}\n")
            f.write(f"noise_mu: {self.noise_mu}\n")
            f.write(f"noise_sigma: {self.noise_sigma}\n")
            f.write(f"u_amp_min: {self.u_amp_min}\n")
            f.write(f"u_amp_max: {self.u_amp_max}\n")
            f.write(f"u_delta_fixed: {self.u_delta_fixed}\n")
            f.write(f"u_delta: {self.u_delta}\n")
            f.write(f"u_delta_min: {self.u_delta_min}\n")
            f.write(f"u_delta_max: {self.u_delta_max}\n")
            f.write(f"u_pause_next: {self.u_pause_next}\n")
            f.write(f"u_density: {self.u_density}\n")
            f.write(f"u_type: {self.u_type}\n\n")

            f.write('Neural Network training parameters:\n')
            f.write(f"PI_ML: {self.PI_ML}\n")
            f.write(f"PI_ML_learn_par: {self.PI_ML_learn_par}\n")
            f.write(f"num_epochs: {self.num_epochs}\n")
            f.write(f"log_iterations: {self.log_iterations}\n")
            f.write(f"eta: {self.eta}\n")
            f.write(f"eta_phys_par: {self.eta_phys_par}\n")
            f.write(f"eta_gain: {self.eta_gain}\n")
            f.write(f"weight_decay: {self.weight_decay}\n")
            f.write(f"test_split_ratio: {self.test_split_ratio}\n")
            f.write(f"val_split_ratio: {self.val_split_ratio}\n")
            f.write(f"bs_train: {self.bs_train}\n")
            f.write(f"bs_val: {self.bs_val}\n")
            f.write(f"bs_test: {self.bs_test}\n")
            f.write(f"init_gain: {self.init_gain}\n")
            f.write(f"start_gain_update: {self.start_gain_update}\n")
            f.write(f"shed_factor: {self.shed_factor}\n")
            f.write(f"shed_cooldown: {self.shed_cooldown}\n")
            f.write(f"shed_patience: {self.shed_patience}\n")            
            f.write(f"es_delta: {self.es_delta}\n")
            f.write(f"es_patience: {self.es_patience}\n")
            f.write(f"phys_dist: {self.phys_dist}\n\n")
            
            if self.architect == 'LF':
                f.write('Learning flow architecture parameters:\n')
                f.write(f"control_rnn_size: {self.control_rnn_size}\n")
                f.write(f"control_rnn_depth: {self.control_rnn_depth}\n")
                f.write(f"enc_dec_depth: {self.enc_dec_depth}\n")
                f.write(f"enc_dec_hidden: {self.enc_dec_hidden}\n\n")
            else:
                f.write('Deep Operator Net architecture parameters:\n')
                f.write(f"n_hidden: {self.n_hidden}\n")
                f.write(f"n_out_branch_trunk: {self.n_out_branch_trunk}\n")
                f.write(f"n_layers: {self.n_layers}\n\n")

            f.write('Physics parameters\n')
            f.write(f"N_r: {self.N_r}\n")
            f.write(f"W: {self.W}\n")
            f.write(f"W_flatten: {self.W_flatten}\n")
            f.write(f"alpha: {self.alpha}\n")
            f.write(f"beta: {self.beta}\n")
            f.write(f"gamma: {self.gamma}\n\n")