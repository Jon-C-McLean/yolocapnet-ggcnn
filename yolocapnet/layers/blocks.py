import torch
from torch import nn
from yolocapnet.layers.conv import Conv

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, shortcut = True):
        super(ConvolutionalBlock, self).__init__()
        out_channels = in_channels // 2
        self.shortcut = shortcut
        self.block1 = Conv(in_channels, out_channels, padding=0)
        self.block2 = Conv(out_channels, in_channels, kernel=3)
    
    def forward(self, x):
        residual = x
        out = self.block1(x)
        out = self.block2(out)

        if self.shortcut:
            out += residual
        
        return out