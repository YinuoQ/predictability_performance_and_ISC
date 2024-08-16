import os
import glob
import math
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset



class PredictAction(Dataset):
    def __init__(self, flag, seed, dataset_folder, time_length):
        super().__init__()
        self.time_out = 30
        self.flag = flag
        self.seed = seed
        self.dataset_folder = dataset_folder
        self.time_length = time_length
        self.current_data = self.get_current_data()
        self.target_seq_len = 30

    def get_current_data(self):
        df = pd.read_csv(os.path.join((self.dataset_folder), f'{self.flag}.csv'), header=None)
        df = df.dropna()
        current_data = df.to_numpy()
        return current_data


    def __len__(self):
        return int(self.current_data.shape[0])

    def __getitem__(self, idx):

        selected_data = self.current_data[idx, :]   
        input_data = selected_data[:-self.time_out]
        output_data = selected_data[-self.time_out:]            

        input_data = torch.Tensor(input_data)
        output_data = torch.FloatTensor(output_data)
        selected_data = torch.tensor(selected_data)
        src, trg, trg_y = self.get_src_trg(
        sequence=selected_data,
        enc_seq_len=selected_data[:-self.time_out].shape[0],
        target_seq_len=selected_data[-self.time_out:].shape[0])
        return src, trg, trg_y

    def get_src_trg(self, sequence: torch.Tensor, enc_seq_len: int, target_seq_len: int):# -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence. 
        Args:
            sequence: tensor, a 1D tensor of length n where 
                    n = encoder input length + target sequence length  
            enc_seq_len: int, the desired length of the input to the transformer encoder
            target_seq_len: int, the desired length of the target sequence (the 
                            one against which the model output is compared)
        Return: 
            src: tensor, 1D, used as input to the transformer model
            trg: tensor, 1D, used as input to the transformer model
            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss. 
        
        """
        # encoder input
        src = sequence[:enc_seq_len] 
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len-1:len(sequence)-1]
        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]
        # We only want trg_y to consist of the target variable not any potential exogenous variables
        return src.float(), trg.float(), trg_y.float() # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] 
    