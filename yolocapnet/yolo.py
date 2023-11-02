from torch import nn
from collections import OrderedDict

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
        return self.activate(self.bn(self.conv(x)))
    
class ResidualBlock(nn.Module): # TODO: Change to dense block
    def __init__(self, in_channels, shortcut = True):
        super(ResidualBlock, self).__init__()
        out_channels = in_channels // 2
        self.shortcut = shortcut
        self.block1 = Conv(in_channels, out_channels, p=0)
        self.block2 = Conv(in_channels, out_channels, k=3)
    
    def forward(self, x):
        residual = x
        out = self.block1(x)
        out = self.block2(out)

        if self.shortcut:
            out += residual
        
        return out

class Darknet53(nn.Module):
    def __init__(self, block, num_class=100, init_weights=True): # Change to allow dev to specify weight initialization func
        super(Darknet53, self).__init__()
        self.num_class = num_class

        if init_weights:
            self.__initialize_weights()

        self.features = nn.Sequential(
            Conv(3, 32, 3),

            Conv(32, 64, 3, 2)
            *self.__get_layer(block, 64, blocks=1),

            Conv(64, 128, 3, 2)
            *self.__get_layer(block, 128, blocks=2),

            Conv(128, 256, 3, 2)
            *self.__get_layer(block, 256, blocks=8),

            Conv(256, 512, 3, 2)
            *self.__get_layer(block, 512, blocks=8),

            Conv(512, 1024, 3, 2)
            *self.__get_layer(block, 1024, blocks=4)
        )

    def forward(self, x):
        return self.features(x)
    
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    @staticmethod
    def __get_layer(block, in_channels, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(in_channels))
        
        return nn.Sequential(*layers)
