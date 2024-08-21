import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from utils import common

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        max_len = 5000
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model) # 1000^(2i/d_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/d_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/d_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return token_embedding + self.pos_encoding[:token_embedding.size(0), :]

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        # Transpose to [seq_len, batch_size, d_model]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # Cross-attention mechanism
        attn_output, _ = self.cross_attn(query, key, value)
        
        # Transpose back to [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(0, 1)
        
        attn_output = self.dropout(attn_output)
        output = self.norm(query + attn_output)  # Residual connection and normalization
        return output


class CrossModalTransformer(nn.Module):
    # Constructor
    def __init__(self):
        super(CrossModalTransformer, self).__init__()
        self.d_model = 64
        self.num_layers = 2
        self.num_head = 4
        self.positional_encoding = PositionalEncoding(self.d_model)
        
        # Convolution layers
        if self.d_model == 32:
            self.conv_eeg = nn.Conv2d(2, 1, (20, 225))
            self.conv_1d = nn.Conv1d(2, 1, 29)
        else:
            self.conv_eeg = nn.Conv2d(2, 1, (20, 193))
            self.conv_1d = nn.Conv1d(2, 1, 1, padding=2)            
        self.tgt_embeding = nn.Linear(30, self.d_model)

        # Cross-attention layers for each modality as query
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttentionLayer(self.d_model, self.num_head) for _ in range(4)
            ]) for _ in range(self.num_layers)
        ])
        
        # Feedforward layers after cross-attention
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, 30),
                nn.ReLU(),
                nn.Linear(30, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.Dropout(0.1)
            ) for _ in range(self.num_layers)
        ])
        
        self.decoder = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.num_head)
        self.classifier = nn.Linear(self.d_model, 90)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src1, src2, src3, src4, tgt):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src1 = self.conv_eeg(src1).squeeze()
        src2 = self.conv_1d(src2).squeeze()
        src3 = self.conv_1d(src3).squeeze()
        src4 = self.conv_1d(src4).squeeze()

        src1 = self.positional_encoding(src1)
        src2 = self.positional_encoding(src2)
        src3 = self.positional_encoding(src3)
        src4 = self.positional_encoding(src4)
        embedded_modalities = [src1, src2, src3, src4]

        # Apply cross-attention layers for each modality as query
        for i, layers in enumerate(self.cross_attention_layers):
            new_embedded_modalities = []
            for j, layer in enumerate(layers):
                # Use modality j as the query and all modalities as key-value pairs
                combined_key_value = torch.cat([embedded_modalities[k] for k in range(len(embedded_modalities))], dim=1)   
                new_embedded_modalities.append(layer(embedded_modalities[j], combined_key_value, combined_key_value))
            embedded_modalities = new_embedded_modalities
  
        # After cross-attention layers, apply feedforward layers
        for feedforward_layer in self.feedforward_layers:
            embedded_modalities = [feedforward_layer(modality) for modality in embedded_modalities]
        
        # Pooling over the sequence for each modality and combine
        pooled_modalities = torch.stack([modality.mean(dim=1) for modality in embedded_modalities], dim=1)  # (N, num_modalities, d_model)
        combined_src = pooled_modalities.mean(dim=1)  # (N, d_model) - Mean pooling across modalities
         
        # Positional encoding for target
        tgt = self.tgt_embeding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer decoder
        decoded_output = self.decoder(tgt.transpose(0, 1), combined_src.unsqueeze(0), tgt_mask=tgt_mask)  # (tgt_len, N, d_model)
        
        # Final classification layer
        output = self.classifier(decoded_output.mean(dim=0))  # (N, num_classes)       
        output = F.softmax(output.view(-1, 3, 30), dim=1)
        return output
        


