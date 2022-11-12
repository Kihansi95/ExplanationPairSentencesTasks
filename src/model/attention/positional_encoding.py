import torch
import math
from torch import Tensor
from torch import nn


class PositionalEncoding(nn.Module):
    """
    This class is made to provide the positional encoding
    originally presented in [Vaswani et al, 2017].

    For the gradient : here it is just a sum. Then there is no problem for the gradient.
    It won't play in the final gradient
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = torch.transpose(x, dim0=0, dim1=1)
        x = x + self.pe[:x.size(0)]
        x = torch.transpose(x, dim0=0, dim1=1)

        return self.dropout(x)
