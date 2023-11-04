import torch
from torch import nn

from yolocapnet.layers.conv import Conv

class Darknet53(nn.Module):
    def __init__(self, block, num_class=10, init_weights=True, num_boxes = 10, include_fc_layer=False): # Change to allow dev to specify weight initialization func
        super(Darknet53, self).__init__()
        self.num_class = num_class
        self.num_boxes = num_boxes
        self.include_fc_layer = include_fc_layer

        if init_weights:
            self.__initialize_weights()

        self.map_1 = nn.Sequential(
            Conv(3, 32, kernel=3),
 
            Conv(32, 64, kernel=3, stride=2),
            *self.__get_layer(block, 64, blocks=1),

            Conv(64, 128, kernel=3, stride=2),
            *self.__get_layer(block, 128, blocks=2),

            Conv(128, 256, kernel=3, stride=2),
            *self.__get_layer(block, 256, blocks=8),
        )

        self.map_2 = nn.Sequential(
            Conv(256, 512, kernel=3, stride=2),
            *self.__get_layer(block, 512, blocks=8),
        )

        self.map_3 = nn.Sequential(
            Conv(512, 1024, kernel=3, stride=2),
            *self.__get_layer(block, 1024, blocks=4),
        )

        self.fc = nn.Linear(1024, self.num_class)

    def forward(self, x):
        route_1 = self.map_1(x)
        route_2 = self.map_2(route_1)
        return route_1, route_2, self.map_3(route_2)

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
        for _ in range(0,blocks):
            layers.append(block(in_channels))

        return nn.Sequential(*layers)