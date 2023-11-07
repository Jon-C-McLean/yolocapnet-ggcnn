import torch
from torch import nn
from .dbl import DBLBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, shortcut = True):
        super(ResidualBlock, self).__init__()
        out_channels = in_channels // 2
        self.shortcut = shortcut
        self.block1 = DBLBlock(in_channels, out_channels, padding=0)
        self.block2 = DBLBlock(out_channels, in_channels, kernel=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.block1(x)
        out = self.block2(out)

        if self.shortcut:
            out += residual
        
        return out