import torch
from torch import nn

def pad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=None, groups=1, activate=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, pad(kernel, padding), groups=groups, bias=False)
        self.normal = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.activate = nn.LeakyReLU(0.01) if activate else nn.ReLU()
    
    def forward(self, x):
        return self.activate(self.normal(self.conv(x)))