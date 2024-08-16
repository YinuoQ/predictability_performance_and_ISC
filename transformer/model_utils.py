import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from utils import common

class MLPLayer(nn.Module):
    def __init__(self, in_f, out_f, if_last=False):
        super().__init__()
        self.in_f = in_f
        self.if_last = if_last
        self.linear = nn.Linear(in_f, out_f)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.linear(x)
        if not self.if_last:
            x = self.relu(x)
        else:
            x = self.softmax(x)
        return x

class TransformerLinearLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_f):
        super().__init__()
        self.num_head = 4

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_f, nhead=self.num_head, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2, norm=None)        

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, in_f):
        super().__init__()
        self.num_head = 4

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=in_f, nhead=self.num_head, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2, norm=None)        
    
    def forward(self, tgt, memory, tgt_mask):
        x = self.transformer_decoder(tgt, memory, tgt_mask)
        return x

class TransformerPositionalEncoder(nn.Module):
    def __init__(self, dropout: float=0.1, max_seq_len: int=5000, d_model: int=64, batch_first: bool=True):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.x_dim = 0 if batch_first else 1

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)      
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        x = x + self.pe[:x.size(self.x_dim)]
        return self.dropout(x)

class ActionPredictionTransformerClassificationModel(torch.nn.Module):
    def __init__(self, in_channels=120, input_modality=['pupil']):
        super(ActionPredictionTransformerClassificationModel, self).__init__()
        self.in_channels = in_channels
        self.input_modality = ['pupil', 'eeg']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.d_model = 64
        
        self.pos_encoder = TransformerPositionalEncoder(d_model=self.d_model)
        self.trans_encoder = TransformerEncoderLayer(self.d_model)
        self.trans_decoder = TransformerDecoderLayer(self.d_model)
        self.decoder_output_layer = TransformerLinearLayer(self.d_model, 3)


    def get_source_data(self, src):
        if 'eeg' not in self.input_modality:
            return src
        else:
            self.processed_eeg = MLPLayer(int(64*20*2), int(64*2)).to(self.device)
            action_location_len = 300
            len_pupil = 60 * 2 if 'pupil' in self.input_modality else 0
            len_eeg = int(64*20*2) if 'eeg' in self.input_modality else 0

            cumulative_len = action_location_len + len_pupil

            eeg_data = self.processed_eeg(src[:, cumulative_len:cumulative_len+len_eeg])
            src_result = torch.cat([src[:, :cumulative_len], eeg_data, src[:, cumulative_len+len_eeg:]], dim=1)
            
            return src_result
    def encoder(self, src):
        # data input layer
        src = self.get_source_data(src)
        src = src.unsqueeze(2).repeat(1, 1, self.d_model)
        src = self.pos_encoder(src)
        src = self.trans_encoder(src)
        return src
    
    def decoder(self, src, trg, memory_mask, tgt_mask):
        trg = trg.unsqueeze(2).repeat(1, 1, self.d_model)
        trg = self.pos_encoder(trg)
        trg = self.trans_decoder(tgt=trg, memory=src, tgt_mask=tgt_mask)
        trg = self.decoder_output_layer(trg)
        return trg


    def forward(self, src, trg, src_mask, trg_mask):
        src = self.encoder(src)
        pred_x = self.decoder(src, trg, src_mask, trg_mask)
        # pass
        return pred_x
        
