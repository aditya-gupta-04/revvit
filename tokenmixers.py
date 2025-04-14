import torch
from torch import nn
import math

# NOT Implemented in the blocks
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, N, C] (code was for B,C,H,W in poolformer)
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

    def forward(self, x):

        x = x.permute(0, 2, 1)  
        x = super().forward(x)

        return x.permute(0, 2, 1)

class PoolingFBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        pool_size,
        patches_shape,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            batch_size, sequence_length, dim = x.shape

            x = self.norm(x)    
            x = x.transpose(1,2).reshape(batch_size, dim, self.patches_shape[0], self.patches_shape[1])
            x = self.pool(x) - x
            x = x.reshape(batch_size, sequence_length, dim)

            return x

class SpatialMLPFBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        patches_shape,
        enable_amp=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.fc = nn.Linear(patches_shape[0]*patches_shape[1], patches_shape[0]*patches_shape[1])
        self.patches_shape = patches_shape
        self.enable_amp = enable_amp

    def forward(self, x):

        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            batch_size, sequence_length, dim = x.shape

            x = self.norm(x)    
            x = x.transpose(1,2)
            x = self.fc(x)
            x = x.transpose(1,2)

            return x