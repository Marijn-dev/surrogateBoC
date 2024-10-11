import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


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
    def __init__(self, raw_data: Dataset_LF_raw, config):
        self.state_dim = config.N_r
        self.control_dim = config.N_r
        self.output_dim = config.N_r
        self.delta = config.u_delta

        init_state = []
        state = []
        rnn_input_data = []
        seq_len_data = []
        t_state = []

        for i, (x0, x0_n, t, y, y_n, u) in tqdm(enumerate(raw_data)):
    
            for k_s, y_s in enumerate(y):
                rnn_input, rnn_input_len = self.process_example(
                    0, k_s, t, u, self.delta)

                init_state.append(x0)
                state.append(y_s)
                seq_len_data.append(rnn_input_len)
                rnn_input_data.append(rnn_input)
                t_state.append(t[k_s])

        self.init_state = torch.stack(init_state).type(
            torch.get_default_dtype())
        self.state = torch.stack(state).type(torch.get_default_dtype())
        
        ## Adjustment made for experimnetal data
        self.rnn_input = torch.nn.utils.rnn.pad_sequence(rnn_input_data, batch_first=True, padding_value=0.0).type(torch.get_default_dtype())

        self.seq_lens = torch.tensor(seq_len_data, dtype=torch.long)
        self.t_state = torch.stack(t_state).type(torch.get_default_dtype())

        self.len = len(init_state)

    @staticmethod
    def process_example(start_idx, end_idx, t, u, delta):
        u_end_idx = int(np.floor(np.round((t[end_idx])/delta,4)))
        u_sz = 1 + u_end_idx 

        u_seq = torch.zeros_like(u)
        u_seq[0:u_sz] = u[0:(u_end_idx + 1)]

        deltas = torch.ones(len(u),1)
        t_u_end = delta * u_end_idx
        t_u_start = 0

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
