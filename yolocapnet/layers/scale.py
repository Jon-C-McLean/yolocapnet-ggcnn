import torch
import torch.nn as nn

from .dbl import DBLBlock

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_class):
        super(ScalePrediction, self).__init__()

        self.prediction = nn.Sequential(
            DBLBlock(in_channels, in_channels * 2, kernel=3, padding=1),
            DBLBlock(2 * in_channels, (num_class + 5) * 3, kernel=1, padding=0)
        )

        self.num_class = num_class

    def forward(self, x):
        return self.prediction(x).reshape(x.shape[0], 3, self.num_class + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)