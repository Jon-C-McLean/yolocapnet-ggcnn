import torch
import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert(x.data.dim() == 4)
        b = x.data.size(0)
        c = x.data.size(1)
        h = x.data.size(2)
        w = x.data.size(3)

        x = x.view(b, c, h, 1, w, 1).expand(b, c, h, self.stride, w, self.stride).contiguous().view(b, c, h*self.stride, w*self.stride)
        return x