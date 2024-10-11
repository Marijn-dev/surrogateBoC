import torch
from torch.utils.data import Dataset
import numpy as np

###################################################################
class Dataset_MLP(Dataset):
    def __init__(self, X_data, y_data):
        self.X = X_data
        self.y = y_data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):    
        X_item = self.X[idx]
        y_item = self.y[idx]

        return X_item, y_item
    
###################################################################
class Dataset_DON(Dataset):
    def __init__(self, X1, X2, X3, y):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3        
        self.y = y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):    
        X_t = self.X1[idx]
        X_u = self.X2[idx]
        X_r0 = self.X3[idx]
        y_item = self.y[idx]
        return X_t, X_u, X_r0, y_item


###################################################################
class Dataset_LF_raw(Dataset):
    def __init__(self, init_state, init_state_noise, time, state, state_noise, control_seq):
        self.init_state = init_state
        self.init_state_noise = init_state_noise
        self.time = time
        self.state = state
        self.state_noise = state_noise
        self.control_seq = control_seq

    def __len__(self):
        return len(self.init_state)

    def __getitem__(self, idx):    
        init_state_item = self.init_state[idx]
        init_state_noise_item = self.init_state_noise[idx]
        time_item = self.time[idx]
        state_item = self.state[idx]
        state_noise_item = self.state_noise[idx]
        control_seq_item = self.control_seq[idx]
        return init_state_item, init_state_noise_item, time_item, state_item, state_noise_item, control_seq_item


class Dataset_LF(Dataset):
    def __init__(self, raw_data: Dataset_LF_raw, config, max_seq_len=-1):
        self.state_dim = config.N_r
        self.control_dim = config.N_r
        self.output_dim = config.N_r
        self.delta = config.u_delta

        init_state = []
        state = []
        rnn_input_data = []
        seq_len_data = []
        t_state = []

        for (x0, x0_n, t, y, y_n, u) in raw_data:
            y += y_n
            x0 += x0_n

            # if max_seq_len == -1:
            for k_s, y_s in enumerate(y):
                rnn_input, rnn_input_len = self.process_example(
                    0, k_s, t, u, self.delta)

                s = y_s.view(1, -1).reshape(-1)

                init_state.append(x0)
                state.append(s)
                seq_len_data.append(rnn_input_len)
                rnn_input_data.append(rnn_input)
                t_state.append(t[k_s])

        self.init_state = torch.stack(init_state).type(
            torch.get_default_dtype())
        self.state = torch.stack(state).type(torch.get_default_dtype())
        self.rnn_input = torch.stack(rnn_input_data).type(
            torch.get_default_dtype())
        self.seq_lens = torch.tensor(seq_len_data, dtype=torch.long)
        self.t_state = torch.stack(t_state).type(torch.get_default_dtype())

        self.len = len(init_state)

    @staticmethod
    def process_example(start_idx, end_idx, t, u, delta):
        init_time = 0.

        u_start_idx = int(np.floor(np.round( (t[start_idx] - init_time)/delta ,2)))
        u_end_idx = int(np.floor(np.round( (t[end_idx] - init_time)/delta ,2)))
        u_sz = 1 + u_end_idx - u_start_idx

        u_seq = torch.zeros_like(u)
        u_seq[0:u_sz] = u[u_start_idx:(u_end_idx + 1)]

        # deltas = torch.ones_like(u_seq)
        deltas = torch.ones(len(u),1)
        t_u_end = init_time + delta * u_end_idx
        t_u_start = init_time + delta * u_start_idx

        if u_sz > 1:
            deltas[0] = (1. - (t[start_idx] - t_u_start) / delta).item()
            deltas[u_sz - 1] = ((t[end_idx] - t_u_end) / delta).item()
        else:
            deltas[0] = ((t[end_idx] - t[start_idx]) / delta).item()

        deltas[u_sz:] = 0.

        rnn_input = torch.hstack((u_seq, deltas))

        return rnn_input, u_sz

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.init_state[index], self.state[index],
                self.rnn_input[index], self.seq_lens[index], self.t_state[index])
