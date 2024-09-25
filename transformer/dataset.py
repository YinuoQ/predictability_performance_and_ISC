import os
import torch
import numpy as np
from torch.utils.data import Dataset

class PredictAction(Dataset):
    def __init__(self, flag, seed, dataset_folder):
        super().__init__()
        self.time_out = 30
        self.flag = flag

        self.seed = seed
        self.dataset_folder = dataset_folder
        self.current_data = self.get_current_data_random()
        self.target_seq_len = 30

    def __len__(self):
        return int(self.current_data[0].shape[0])

    def __getitem__(self, idx):
        eeg, pupil, speech, action, location, out_action = self.current_data
        selected_eeg = eeg[idx]
        selected_pupil = pupil[idx]
        selected_speech = speech[idx]
        selected_action = action[idx]
        selected_location = location[idx]
        selected_out_action = out_action[idx]

        input_eeg = torch.Tensor(selected_eeg)
        input_pupil = torch.Tensor(selected_pupil)
        input_speech = torch.Tensor(selected_speech)
        input_action = torch.Tensor(selected_action)
        input_location = torch.Tensor(selected_location)
        output_data = torch.Tensor(selected_out_action)

        src1, src2, src3, src4, src5, tgt, trg_y = self.get_src_trg(
        sequence1=input_eeg,
        sequence2=input_pupil,
        sequence3=input_speech,
        sequence4=input_action,
        sequence5=input_location,
        output_seq=output_data)
        
        return src1, src2, src3, src4, src5, tgt, trg_y
    
    def get_current_data(self):
        eeg = np.load(os.path.join((self.dataset_folder), f'{self.flag}', f'{self.flag}_eeg.npy'), allow_pickle=True)
        pupil = np.load(os.path.join((self.dataset_folder), f'{self.flag}', f'{self.flag}_pupil.npy'), allow_pickle=True)
        speech = np.load(os.path.join((self.dataset_folder), f'{self.flag}', f'{self.flag}_speech.npy'), allow_pickle=True)
        action = np.load(os.path.join((self.dataset_folder), f'{self.flag}', f'{self.flag}_action.npy'), allow_pickle=True)
        location = np.load(os.path.join((self.dataset_folder), f'{self.flag}', f'{self.flag}_location.npy'), allow_pickle=True)
        out_action = np.load(os.path.join((self.dataset_folder), f'{self.flag}', f'{self.flag}_output.npy'), allow_pickle=True)
   
        return eeg, pupil, speech, action, location, out_action
    
    def get_current_data_random(self):
        eeg = np.load(os.path.join((self.dataset_folder), f'{self.flag}_random', f'{self.flag}_eeg.npy'), allow_pickle=True)
        pupil = np.load(os.path.join((self.dataset_folder), f'{self.flag}_random', f'{self.flag}_pupil.npy'), allow_pickle=True)
        speech = np.load(os.path.join((self.dataset_folder), f'{self.flag}_random', f'{self.flag}_speech.npy'), allow_pickle=True)
        action = np.load(os.path.join((self.dataset_folder), f'{self.flag}_random', f'{self.flag}_action.npy'), allow_pickle=True)
        location = np.load(os.path.join((self.dataset_folder), f'{self.flag}_random', f'{self.flag}_location.npy'), allow_pickle=True)
        out_action = np.load(os.path.join((self.dataset_folder), f'{self.flag}_random', f'{self.flag}_output.npy'), allow_pickle=True)

        return eeg, pupil, speech, action, location, out_action

    def get_src_trg(self, sequence1: torch.Tensor, sequence2: torch.Tensor, sequence3: torch.Tensor, sequence4: torch.Tensor, sequence5: torch.Tensor, output_seq: torch.Tensor):

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
        src1 = sequence1
        src2 = sequence2
        src3 = sequence3
        src4 = sequence4
        src5 = sequence5

        # The target sequence against which the model output will be compared to compute loss   
        # if self.flag != 'test':
        #     tgt = torch.cat((output_seq[:1], output_seq[:-1]), dim=0)  # Shifting output_seq
        # else:
        #     tgt = torch.cat((output_seq[:1], torch.zeros_like(output_seq[:-1])), dim=0)

        tgt = output_seq[:self.target_seq_len]
        trg_y = output_seq[self.target_seq_len:]
        # We only want trg_y to consist of the target variable not any potential exogenous variables
        return src1, src2, src3, src4, src5, tgt, trg_y