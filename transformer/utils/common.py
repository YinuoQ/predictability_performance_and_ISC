import os
import shutil
import torch
from torch import nn, Tensor

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def generate_square_subsequent_mask(dim0: int, dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Source:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim0: int, batch size * num_heads (2)
        dim1: int, for both src and tgt masking, this must be target sequence
                length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
                the length of the input sequence to the model), 
                and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim0, dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1).repeat(dim0, 1, 1)
