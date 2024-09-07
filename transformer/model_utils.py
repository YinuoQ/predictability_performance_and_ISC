import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from utils import common

class AdaptiveConvLayer(nn.Module):
    def __init__(self, in_chenn_len, out_chann):
        super(AdaptiveConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_chenn_len, out_chann, kernel_size=1, stride=1, padding=2)
        self.conv_loc = nn.Conv1d(in_chenn_len, out_chann, kernel_size=2, stride=1, padding=2)
        self.conv_tgt = nn.Conv1d(in_chenn_len, out_chann, kernel_size=1, stride=1, padding=1)
    
    def forward(self, x):
        # Flatten the time dimension if necessary
        if x.shape[2] == 60:
            x = self.conv(x)
        elif x.shape[2] == 30:
            x = self.conv_tgt(x)
        else:
            x = self.conv_loc(x)
        return x.permute(0,2,1)

class AdaptiveConv2dLayer(nn.Module):
    def __init__(self, out_chann):
        super(AdaptiveConv2dLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=out_chann, kernel_size=(20, 2), stride=(1, 4))

    def forward(self, x):
        x = self.conv2d(x)  # Apply 2D convolution
        x = x.squeeze(2)  # Remove the singleton dimension after convolution
        return x.permute(0,2,1)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, x.size(1), :].to(x.device)
        
class CrossModalTransformer(nn.Module):
    def __init__(self, num_classes=3, time_steps=30):
        super(CrossModalTransformer, self).__init__()
        self.num_heads = 8
        self.conv_output_dim = 8
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Adaptive convolutional layers for each modality
        self.eeg_conv = AdaptiveConv2dLayer(self.conv_output_dim)
        self.pupil_speech_action_conv = AdaptiveConvLayer(2, self.conv_output_dim)
        self.location_conv = AdaptiveConvLayer(3, self.conv_output_dim)
        self.tgt_conv = AdaptiveConvLayer(1, self.conv_output_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.conv_output_dim)
        self.tgt_pos_encoder = PositionalEncoding(self.conv_output_dim)

        # normalization
        self.norm = nn.LayerNorm(self.conv_output_dim)
        self.tgt_norm = nn.LayerNorm(self.conv_output_dim*4)
       
        # Cross-modal attention layers (self-attention for query/value and cross-attention for key)
        self.cross_attention = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.4)
        self.self_attention = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.tgt_attention = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.output_attention = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)


        # Final layers
        self.fc1 = nn.Linear(in_features=256, out_features=90)

    def generate_subsequent_mask(self, sz1, sz2):
        mask = (torch.triu(torch.ones(sz2, sz1)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, eeg, pupil, speech, action, location, tgt):
        # Apply adaptive convolution
        eeg = self.eeg_conv(eeg)
        pupil = self.pupil_speech_action_conv(pupil)
        speech = self.pupil_speech_action_conv(speech)
        action = self.pupil_speech_action_conv(action)
        location = self.location_conv(location)

        # Apply positional encoding
        eeg = self.pos_encoder(eeg)
        pupil = self.pos_encoder(pupil)
        speech = self.pos_encoder(speech)
        action = self.pos_encoder(action)
        location = self.pos_encoder(location)

        # normalizing 
        # eeg = self.norm(eeg)
        # pupil = self.norm(pupil)
        # speech = self.norm(speech)
        # action = self.norm(action)
        # location = self.norm(location)
        
        # Cross-modality attention
        eeg_cross_pupil = self.cross_attention(eeg, pupil, pupil, need_weights=False)[0]
        eeg_cross_speech = self.cross_attention(eeg, speech, speech, need_weights=False)[0]
        eeg_cross_action = self.cross_attention(eeg, action, action, need_weights=False)[0]
        eeg_cross_location = self.cross_attention(eeg, location, location, need_weights=False)[0]
        eeg_self = self.self_attention(eeg, eeg, eeg, need_weights=False)[0]
        eeg_cross_pupil = self.norm(eeg_cross_pupil)
        eeg_cross_speech = self.norm(eeg_cross_speech)
        eeg_cross_action = self.norm(eeg_cross_action)
        eeg_cross_location = self.norm(eeg_cross_location)
        eeg_self = self.norm(eeg_self)

        pupil_cross_eeg = self.cross_attention(pupil, eeg, eeg, need_weights=False)[0]
        pupil_cross_speech = self.cross_attention(pupil, speech, speech, need_weights=False)[0]
        pupil_cross_action = self.cross_attention(pupil, action, action, need_weights=False)[0]
        pupil_cross_location = self.cross_attention(pupil, location, location, need_weights=False)[0]
        pupil_self = self.self_attention(pupil, pupil, pupil, need_weights=False)[0]
        
        pupil_cross_eeg = self.norm(pupil_cross_eeg)
        pupil_cross_speech = self.norm(pupil_cross_speech)
        pupil_cross_action = self.norm(pupil_cross_action)
        pupil_cross_location = self.norm(pupil_cross_location)
        pupil_self = self.norm(pupil_self)

        speech_cross_eeg = self.cross_attention(speech, eeg, eeg, need_weights=False)[0]
        speech_cross_pupil = self.cross_attention(speech, pupil, pupil, need_weights=False)[0]
        speech_cross_action = self.cross_attention(speech, action, action, need_weights=False)[0]
        speech_cross_location = self.cross_attention(speech, location, location, need_weights=False)[0]
        speech_self = self.self_attention(speech, speech, speech, need_weights=False)[0]

        speech_cross_eeg = self.norm(speech_cross_eeg)
        speech_cross_pupil = self.norm(speech_cross_pupil)
        speech_cross_action = self.norm(speech_cross_action)
        speech_cross_location = self.norm(speech_cross_location)
        speech_self = self.norm(speech_self)


        action_cross_eeg = self.cross_attention(action, eeg, eeg, need_weights=False)[0]
        action_cross_pupil = self.cross_attention(action, pupil, pupil, need_weights=False)[0]
        action_cross_speech = self.cross_attention(action, speech, speech, need_weights=False)[0]
        action_cross_location = self.cross_attention(action, location, location, need_weights=False)[0]
        action_self = self.self_attention(action, action, action, need_weights=False)[0]
       
        action_cross_eeg = self.norm(action_cross_eeg)
        action_cross_pupil = self.norm(action_cross_pupil)
        action_cross_speech = self.norm(action_cross_speech)
        action_cross_location = self.norm(speech_cross_location)
        action_self = self.norm(action_self)

        # Sum the outputs of cross-attention for each modality
        eeg_final = eeg_cross_pupil + eeg_cross_action + eeg_cross_location + eeg_self + eeg_cross_speech
        pupil_final = pupil_cross_eeg + pupil_cross_action + pupil_cross_location + pupil_self + pupil_cross_speech
        speech_final = speech_cross_eeg + speech_cross_pupil + speech_cross_action + speech_cross_location
        action_final = action_cross_eeg + action_cross_pupil + action_cross_location + action_self + action_cross_speech
        
        # Concatenate modalities
        concatenated = torch.cat([eeg_final, pupil_final, action_final, speech_final], dim=-2)#, speech_self
        # import IPython
        # IPython.embed()
        # assert False
        
        # should remove tgt encoder here
        tgt = self.tgt_conv(tgt.unsqueeze(1))
        tgt = self.tgt_pos_encoder(tgt)  # Apply positional encoding to the target
        tgt_mask = self.generate_subsequent_mask(tgt.size(0), tgt.size(1)).to(self.device)
        # tgt = self.tgt_attention(tgt, tgt, tgt, key_padding_mask=tgt_mask)[0]
        output = self.output_attention(tgt, concatenated, concatenated)[0]
        # output = self.tgt_norm(output)


        # Final output layer
        output = self.fc1(torch.reshape(output, (-1, output.shape[1] * output.shape[2])))
        output = torch.transpose(output.view(-1, 30, 3), 1,2)
        output = torch.softmax(output, dim=1)
        return output