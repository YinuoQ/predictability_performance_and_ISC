import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from utils import common

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         """Positional Encoding.

#         Args:
#             d_model: Hidden dimensionality of the input.
#             max_len: Maximum length of a sequence to expect.

#         """
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = self.encoding.unsqueeze(0)
        
#     def forward(self, x):
#         seq_len = x.size(1)
#         return x + self.encoding[:, :seq_len, :].to(x.device)
    
    
# class CrossAttentionLayer(nn.Module):
#     def __init__(self, d_model, nhead, dropout=0.1):
#         super(CrossAttentionLayer, self).__init__()
#         self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
#         self.norm = nn.LayerNorm(d_model)
    
#     def forward(self, query, key, value):
#         # Cross-attention mechanism
#         attn_output, _ = self.cross_attn(query, key, value)
        
#         output = self.norm(query + attn_output)  # Residual connection and normalization
#         return output


# class CrossModalTransformer(nn.Module):
#     # Constructor
#     def __init__(self):
#         super(CrossModalTransformer, self).__init__()
#         self.d_model = 64
#         self.num_layers = 1
#         self.num_head = 8
#         self.positional_encoding = PositionalEncoding(self.d_model)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Convolution layers
#         if self.d_model == 32:
#             self.conv_eeg = nn.Conv2d(2, 1, (20, 225))
#             self.conv_1d = nn.Conv1d(2, 1, 29)
#         else:
#             self.conv_eeg = nn.Conv2d(2, 1, (20, 193))
#             self.conv_1d = nn.Conv1d(2, 1, 1, padding=2)            

#         # Cross-attention layers for each modality as query
#         self.cross_attention_layers = nn.ModuleList([
#             nn.ModuleList([
#                 CrossAttentionLayer(self.d_model, self.num_head) for _ in range(4)
#             ]) for _ in range(self.num_layers)
#         ])
#         # Self-attention layers for each modality separately
#         self.self_attention_layers = nn.ModuleList([
#             nn.MultiheadAttention(self.d_model, self.num_head, dropout=0.1, batch_first=True) for _ in range(4)
#         ])

#         self.classifier = nn.Sequential(
#                 nn.Linear(16*self.d_model, 512),
#                 nn.BatchNorm1d(512), #applying batch norm
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(512, 256),
#                 nn.BatchNorm1d(256), #applying batch norm
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(256, 90),
#                 nn.BatchNorm1d(90))
    
#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
        
#     def forward(self, src1, src2, src3, src4, tgt):

#         # Src size must be (batch_size, src sequence length)
#         # Tgt size must be (batch_size, tgt sequence length)
#         # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
#         src1 = self.conv_eeg(src1).squeeze(dim=1)
#         src2 = self.conv_1d(src2)
#         src3 = self.conv_1d(src3)
#         src4 = self.conv_1d(src4)

#         src1 = self.positional_encoding(src1)
#         src2 = self.positional_encoding(src2)
#         src3 = self.positional_encoding(src3)
#         src4 = self.positional_encoding(src4)
#         embedded_modalities = [src1, src2, src3, src4]
#         # embedded_modalities = [src1]


#         # Apply cross-attention layers for each modality as query
        
#         for layer_set in self.cross_attention_layers:
#             new_embedded_modalities = []
#             # A -> B, then A -> C, then A -> D
#             temp_embedded_modalities = []
#             for layer_idx, modality_idx in enumerate([1, 2, 3]):
#                 temp_embedded_modalities.append(layer_set[0](embedded_modalities[0], embedded_modalities[modality_idx], embedded_modalities[modality_idx]))
#             new_embedded_modalities.append(torch.cat(temp_embedded_modalities, dim=1))
#             # B -> A, then B -> C, then B -> D
#             temp_embedded_modalities = []
#             for layer_idx, modality_idx in enumerate([0, 2, 3]):
#                 temp_embedded_modalities.append(layer_set[1](embedded_modalities[1], embedded_modalities[modality_idx], embedded_modalities[modality_idx]))
#             new_embedded_modalities.append(torch.cat(temp_embedded_modalities, dim=1))
#             # C -> A, then C -> B, then C -> D
#             temp_embedded_modalities = []
#             for layer_idx, modality_idx in enumerate([0, 1, 3]):
#                 temp_embedded_modalities.append(layer_set[2](embedded_modalities[2], embedded_modalities[modality_idx], embedded_modalities[modality_idx]))
#             new_embedded_modalities.append(torch.cat(temp_embedded_modalities, dim=1))
#             # D -> A, then D -> B, then D -> C
#             temp_embedded_modalities = []
#             for layer_idx, modality_idx in enumerate([0, 1, 2]):
#                 temp_embedded_modalities.append(layer_set[3](embedded_modalities[3], embedded_modalities[modality_idx], embedded_modalities[modality_idx]))
#             new_embedded_modalities.append(torch.cat(temp_embedded_modalities, dim=1))
        
#         # Apply self-attention layers for each modality separately
#         for i in range(4):
#             modality = embedded_modalities[i] # Shape [batch_size, seq_len, d_model]
#             modal_size = modality.size(1)
#             src_mask = self.generate_square_subsequent_mask(modal_size).to(self.device)
#             modality, _ = self.self_attention_layers[i](modality, modality, modality, attn_mask=src_mask)
#             new_embedded_modalities[i] = torch.cat([new_embedded_modalities[i], modality], dim=1)  # Back to [batch_size, seq_len, d_model]
   
#         # Pooling over the sequence for each modality and combine
#         pooled_modalities = torch.cat([modality for modality in new_embedded_modalities], dim=1)  # (N, num_modalities, d_model)

#         # Final classification layer
#         output = self.classifier(pooled_modalities.view(-1, 16*self.d_model))  # (N, num_classes)     
#         output = torch.softmax(output.view(-1, 3, 30), dim=1)
    
#         return output
    

class AdaptiveConvLayer(nn.Module):
    def __init__(self):
        super(AdaptiveConvLayer, self).__init__()
        self.conv = nn.Conv1d(2, 64, kernel_size=1)
    
    def forward(self, x):
        # Flatten the time dimension if necessary
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x

class AdaptiveConv2dLayer(nn.Module):
    def __init__(self):
        super(AdaptiveConv2dLayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(20, 20), stride=(20, 4))

    def forward(self, x):
        x = self.conv2d(x)  # Apply 2D convolution
        x = x.squeeze(2)  # Remove the singleton dimension after convolution
        x = x.permute(0, 2, 1)

        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class CrossModalTransformer(nn.Module):
    def __init__(self, num_classes=3, time_steps=30):
        super(CrossModalTransformer, self).__init__()
        self.num_heads = 4
        self.conv_output_dim = 64
        # Adaptive convolutional layers for each modality
        self.eeg_conv = AdaptiveConv2dLayer()
        self.pupil_conv = AdaptiveConvLayer()
        self.speech_conv = AdaptiveConvLayer()
        self.action_conv = AdaptiveConvLayer()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.conv_output_dim)
       
        # Cross-modal attention layers (self-attention for query/value and cross-attention for key)
        self.cross_attention_layers = nn.ModuleDict({
            'eeg_to_pupil': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'eeg_to_speech': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'eeg_to_action': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'pupil_to_eeg': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'pupil_to_speech': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'pupil_to_action': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'speech_to_eeg': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'speech_to_pupil': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'speech_to_action': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'action_to_eeg': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'action_to_pupil': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'action_to_speech': nn.MultiheadAttention(self.conv_output_dim, self.num_heads)
        })
        
        # Self-attention layers for each modality
        self.self_attention_layers = nn.ModuleDict({
            'eeg_self': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'pupil_self': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'speech_self': nn.MultiheadAttention(self.conv_output_dim, self.num_heads),
            'action_self': nn.MultiheadAttention(self.conv_output_dim, self.num_heads)
        })
        # Final layers
        self.fc = nn.Linear(in_features=15360, out_features=90)
        self.num_classes = num_classes
        self.time_steps = time_steps
    
    def forward(self, eeg, pupil, speech, action):
        # Apply adaptive convolution
        eeg = self.eeg_conv(eeg)
        pupil = self.pupil_conv(pupil)
        speech = self.speech_conv(speech)
        action = self.action_conv(action)

        # Apply positional encoding
        eeg = self.pos_encoder(eeg)
        pupil = self.pos_encoder(pupil)
        speech = self.pos_encoder(speech)
        action = self.pos_encoder(action)
        
        # Cross-modality attention
        eeg_cross_pupil, _ = self.cross_attention_layers['eeg_to_pupil'](eeg, pupil, pupil)
        eeg_cross_speech, _ = self.cross_attention_layers['eeg_to_speech'](eeg, speech, speech)
        eeg_cross_action, _ = self.cross_attention_layers['eeg_to_action'](eeg, action, action)
        
        pupil_cross_eeg, _ = self.cross_attention_layers['pupil_to_eeg'](pupil, eeg, eeg)
        pupil_cross_speech, _ = self.cross_attention_layers['pupil_to_speech'](pupil, speech, speech)
        pupil_cross_action, _ = self.cross_attention_layers['pupil_to_action'](pupil, action, action)
        
        speech_cross_eeg, _ = self.cross_attention_layers['speech_to_eeg'](speech, eeg, eeg)
        speech_cross_pupil, _ = self.cross_attention_layers['speech_to_pupil'](speech, pupil, pupil)
        speech_cross_action, _ = self.cross_attention_layers['speech_to_action'](speech, action, action)
        
        action_cross_eeg, _ = self.cross_attention_layers['action_to_eeg'](action, eeg, eeg)
        action_cross_pupil, _ = self.cross_attention_layers['action_to_pupil'](action, pupil, pupil)
        action_cross_speech, _ = self.cross_attention_layers['action_to_speech'](action, speech, speech)
        
        
        # Sum the outputs of cross-attention for each modality
        eeg_final = eeg_cross_pupil + eeg_cross_speech + eeg_cross_action
        pupil_final = pupil_cross_eeg + pupil_cross_speech + pupil_cross_action
        speech_final = speech_cross_eeg + speech_cross_pupil + speech_cross_action
        action_final = action_cross_eeg + action_cross_pupil + action_cross_speech
        
        # Self-attention within each modality
        eeg_self, _ = self.self_attention_layers['eeg_self'](eeg_final, eeg_final, eeg_final)
        pupil_self, _ = self.self_attention_layers['pupil_self'](pupil_final, pupil_final, pupil_final)
        speech_self, _ = self.self_attention_layers['speech_self'](speech_final, speech_final, speech_final)
        action_self, _ = self.self_attention_layers['action_self'](action_final, action_final, action_final)
        
        # Concatenate modalities
        concatenated = torch.cat([eeg_self, pupil_self, speech_self, action_self], dim=-1)

        # Final output layer
        output = self.fc(concatenated.view(eeg.shape[0], -1))
        output = output.view(-1, 3, 30)
        
        return output