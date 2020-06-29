import torch.nn as nn
import torch.nn.functional as F

class nnNorm(nn.Module):
    '''Normalisation layer.'''

    def __init__(self, dim=-1):
        '''Normalises a vector (or batch of vectors) to have unit-(L2)-norm.

        Args:
            dim: The dimension along which to normalise.
        '''
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, dim=self.dim)