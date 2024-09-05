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
    
    def forward(self, x):
        # Flatten the time dimension if necessary
        if x.shape[2] == 60:
            x = self.conv(x)
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
        self.num_heads = 4
        self.conv_output_dim = 8
        # Adaptive convolutional layers for each modality
        self.eeg_conv = AdaptiveConv2dLayer(self.conv_output_dim)
        self.pupil_conv = AdaptiveConvLayer(2, self.conv_output_dim)
        self.speech_conv = AdaptiveConvLayer(2, self.conv_output_dim)
        self.action_conv = AdaptiveConvLayer(2, self.conv_output_dim)
        self.location_conv = AdaptiveConvLayer(3, self.conv_output_dim)
        self.tgt_conv = AdaptiveConvLayer(1, self.conv_output_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.conv_output_dim)
        self.tgt_encoder = PositionalEncoding(self.conv_output_dim*4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # normalization
        self.norm = nn.LayerNorm(self.conv_output_dim)
        self.norm2 = nn.LayerNorm(3840)
       
        # Cross-modal attention layers (self-attention for query/value and cross-attention for key)
        self.ca_eeg_2_pp = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_eeg_2_sp = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_eeg_2_ac = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_eeg_2_lo = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)

        self.ca_pp_2_eeg = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_pp_2_sp = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_pp_2_ac = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_pp_2_lo = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
 
        self.ca_sp_2_eeg = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_sp_2_pp = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_sp_2_ac = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_sp_2_lo = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        
        self.ca_ac_2_eeg = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_ac_2_pp = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_ac_2_sp = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
        self.ca_ac_2_lo = nn.MultiheadAttention(self.conv_output_dim, self.num_heads, batch_first=True, dropout=0.2)
                               
        # Self-attention layers for each modality
        self.sa_eeg = nn.MultiheadAttention(int(self.conv_output_dim), self.num_heads, batch_first=True, dropout=0.2)
        self.sa_pp = nn.MultiheadAttention(int(self.conv_output_dim), self.num_heads, batch_first=True, dropout=0.2)
        self.sa_sp = nn.MultiheadAttention(int(self.conv_output_dim), self.num_heads, batch_first=True, dropout=0.2)
        self.sa_ac = nn.MultiheadAttention(int(self.conv_output_dim), self.num_heads, batch_first=True, dropout=0.2)
        
        # Final layers
        self.fc1 = nn.Linear(in_features=960, out_features=90)
        # self.fc2 = nn.Linear(in_features=256, out_features=3*time_steps)
        self.num_classes = num_classes
        self.time_steps = time_steps
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.conv_output_dim*4, nhead=self.num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, eeg, pupil, speech, action, location, tgt):
        # Apply adaptive convolution

        eeg = self.eeg_conv(eeg)
        pupil = self.pupil_conv(pupil)
        speech = self.speech_conv(speech)
        action = self.action_conv(action)
        location = self.location_conv(location)

        # Apply positional encoding
        eeg = self.pos_encoder(eeg)
        pupil = self.pos_encoder(pupil)
        speech = self.pos_encoder(speech)
        action = self.pos_encoder(action)
        location = self.pos_encoder(location)

        # normalizing 
        eeg = self.norm(eeg)
        pupil = self.norm(pupil)
        speech = self.norm(speech)
        action = self.norm(action)
        location = self.norm(location)
        
        key_padding_mask = self.generate_square_subsequent_mask(64).to(self.device)
        # Cross-modality attention
        eeg_cross_pupil, _ = self.ca_eeg_2_pp(eeg, pupil, pupil, attn_mask=key_padding_mask)
        # eeg_cross_pupil = self.norm(eeg_cross_pupil)
        eeg_cross_speech, _ = self.ca_eeg_2_sp(eeg, speech, speech, attn_mask=key_padding_mask)
        # eeg_cross_speech = self.norm(eeg_cross_speech)
        eeg_cross_action, _ = self.ca_eeg_2_ac(eeg, action, action, attn_mask=key_padding_mask)
        # eeg_cross_action = self.norm(eeg_cross_action)
        eeg_cross_location, _ = self.ca_eeg_2_lo(eeg, location, location, attn_mask=key_padding_mask)
        # eeg_cross_location = self.norm(eeg_cross_location)
        
        pupil_cross_eeg, _ = self.ca_pp_2_eeg(pupil, eeg, eeg, attn_mask=key_padding_mask)
        # pupil_cross_eeg = self.norm(pupil_cross_eeg)
        pupil_cross_speech, _ = self.ca_pp_2_sp(pupil, speech, speech, attn_mask=key_padding_mask)
        # pupil_cross_speech = self.norm(pupil_cross_speech)
        pupil_cross_action, _ = self.ca_pp_2_ac(pupil, action, action, attn_mask=key_padding_mask)
        # pupil_cross_action = self.norm(pupil_cross_action)
        pupil_cross_location, _ = self.ca_pp_2_lo(pupil, location, location, attn_mask=key_padding_mask)
        # pupil_cross_location = self.norm(pupil_cross_location)

        
        speech_cross_eeg, _ = self.ca_sp_2_eeg(speech, eeg, eeg, attn_mask=key_padding_mask)
        # speech_cross_eeg = self.norm(speech_cross_eeg)
        speech_cross_pupil, _ = self.ca_sp_2_pp(speech, pupil, pupil, attn_mask=key_padding_mask)
        # speech_cross_pupil = self.norm(speech_cross_pupil)
        speech_cross_action, _ = self.ca_sp_2_ac(speech, action, action, attn_mask=key_padding_mask)
        # speech_cross_action = self.norm(speech_cross_action)
        speech_cross_location, _ = self.ca_sp_2_lo(speech, location, location, attn_mask=key_padding_mask)
        # speech_cross_location = self.norm(speech_cross_location)

        
        action_cross_eeg, _ = self.ca_ac_2_eeg(action, eeg, eeg, attn_mask=key_padding_mask)
        # action_cross_eeg = self.norm(action_cross_eeg)
        action_cross_pupil, _ = self.ca_ac_2_pp(action, pupil, pupil, attn_mask=key_padding_mask)
        # action_cross_pupil = self.norm(action_cross_pupil)
        action_cross_speech, _ = self.ca_ac_2_sp(action, speech, speech, attn_mask=key_padding_mask)
        # action_cross_speech = self.norm(action_cross_speech)
        action_cross_location, _ = self.ca_ac_2_lo(action, location, location, attn_mask=key_padding_mask)
        # action_cross_location = self.norm(action_cross_location)

        # Sum the outputs of cross-attention for each modality
        eeg_final = eeg_cross_pupil + eeg_cross_speech + eeg_cross_action + eeg_cross_location
        pupil_final = pupil_cross_eeg + pupil_cross_speech + pupil_cross_action + pupil_cross_location
        speech_final = speech_cross_eeg + speech_cross_pupil + speech_cross_action + speech_cross_location
        action_final = action_cross_eeg + action_cross_pupil + action_cross_speech + action_cross_location
        # eeg_final = torch.cat([eeg_cross_pupil, eeg_cross_speech, eeg_cross_action, eeg_cross_location], axis=2)
        # pupil_final = torch.cat([pupil_cross_eeg , pupil_cross_speech , pupil_cross_action , pupil_cross_location], axis=2)
        # speech_final = torch.cat([speech_cross_eeg , speech_cross_pupil , speech_cross_action, speech_cross_location], axis=2)
        # action_final = torch.cat([action_cross_eeg , action_cross_pupil , action_cross_speech , action_cross_location], axis=2)
            
        # Self-attention within each modality
        eeg_self, _ = self.sa_eeg(eeg_final, eeg_final, eeg_final)
        # eeg_self = self.norm(eeg_self)
        pupil_self, _ = self.sa_pp(pupil_final, pupil_final, pupil_final)
        # pupil_self = self.norm(pupil_self)
        speech_self, _ = self.sa_sp(speech_final, speech_final, speech_final)
        # speech_self = self.norm(speech_self)
        action_self, _ = self.sa_ac(action_final, action_final, action_final)
        # action_self = self.norm(action_self)
        
        # Concatenate modalities
        concatenated = torch.cat([eeg_self, pupil_self, speech_self, action_self], dim=-1)
        
        # should remove tgt encoder here
        tgt = self.tgt_encoder(tgt[:,:,None])  # Apply positional encoding to the target
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
        output = self.decoder(tgt, concatenated, tgt_mask=tgt_mask)

        # Final output layer
        output = self.fc1(output.view(eeg.shape[0], -1))
        # output = self.fc2(output)
        output = output.view(eeg.shape[0], 3, 30)
 
        return output