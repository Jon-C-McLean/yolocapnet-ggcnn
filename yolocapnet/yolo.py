import torch
from torch import nn

from layers.dbl import DBLBlock
from layers.blocks import ResidualBlock
from layers.misc import EmptyLayer
from layers.scale import ScalePrediction

class Darknet53(nn.Module):
    def __init__(self, block, num_class=10, init_weights=True, num_boxes = 10): # Change to allow dev to specify weight initialization func
        super(Darknet53, self).__init__()
        self.num_class = num_class
        self.num_boxes = num_boxes

        if init_weights:
            self.__initialize_weights()

        self.map_1 = nn.Sequential(
            DBLBlock(3, 32, kernel=3, padding=1),
 
            DBLBlock(32, 64, kernel=3, stride=2, padding=1),
            *self.__get_layer(block, 64, blocks=1),

            DBLBlock(64, 128, kernel=3, stride=2, padding=1),
            *self.__get_layer(block, 128, blocks=2),

            DBLBlock(128, 256, kernel=3, stride=2, padding=1),
            *self.__get_layer(block, 256, blocks=8),
            # EmptyLayer(),
        )

        self.map_2 = nn.Sequential(
            DBLBlock(256, 512, kernel=3, stride=2, padding=1),
            *self.__get_layer(block, 512, blocks=8),
            # EmptyLayer(),
        )

        self.map_3 = nn.Sequential(
            DBLBlock(512, 1024, kernel=3, stride=2, padding=1),
            *self.__get_layer(block, 1024, blocks=4),
            # EmptyLayer(),
        )

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

class YOLOv3(nn.Module):
    def __init__(self, num_class=10):
        super(YOLOv3, self).__init__()
        self.num_class = num_class

        self.backbone = Darknet53(block=ResidualBlock, num_class=num_class, init_weights=True)

        # (512, 1, 1),
        # (1024, 3, 1),
        # "S",
        # (256, 1, 1),
        # "U",
        # (256, 1, 1),
        # (512, 3, 1),
        # "S",
        # (128, 1, 1),
        # "U",
        # (128, 1, 1),
        # (256, 3, 1),
        # "S",

        self.ldbl = nn.Sequential(
            DBLBlock(1024, 512, kernel=1),
            DBLBlock(512, 1024, kernel=3, padding=1),
        )
        self.lppred = nn.Sequential(
            ResidualBlock(1024, shortcut=False),
            DBLBlock(1024, 512, kernel=1, padding=0),
        )
        self.lspred = ScalePrediction(512, num_class)
    
        self.ups_1 = nn.Sequential(
            DBLBlock(512, 256, kernel=1, padding=0),
            nn.Upsample(scale_factor=2),
        )
        self.mdbl = nn.Sequential(
            DBLBlock(256*3, 256, kernel=1, padding=0),
            DBLBlock(256, 512, kernel=3, padding=1),
        )
        self.mppred = nn.Sequential(
            ResidualBlock(512, shortcut=False),
            DBLBlock(512, 256, kernel=1, padding=0),
        )
        self.mspred = ScalePrediction(256, num_class)

        self.ups_2 = nn.Sequential(
            DBLBlock(256, 128, kernel=1),
            nn.Upsample(scale_factor=2),
        )
        self.sdbl = nn.Sequential(
            DBLBlock(128*3, 128, kernel=1, padding=0),
            DBLBlock(128, 256, kernel=3, padding=1),
        )
        self.sppred = nn.Sequential(
            ResidualBlock(256, shortcut=False),
            DBLBlock(256, 128, kernel=1, padding=0),
        )
        self.sspred = ScalePrediction(128, num_class)

    def forward(self, x): # XXX: Do I need to add mbox layers?
        route_1, route_2, dn_out = self.backbone(x)
        out = self.ldbl(dn_out)
        out = self.lppred(out)

        self.l_pred = self.lspred(out)

        out = self.ups_1(out)
        out = torch.cat([out, route_2], dim=1)
        out = self.mdbl(out)
        out = self.mppred(out)

        self.m_pred = self.mspred(out)

        out = self.ups_2(out)
        out = torch.cat([out, route_1], dim=1)
        out = self.sdbl(out)
        out = self.sppred(out)

        self.s_pred = self.sspred(out)

        return [self.l_pred, self.m_pred, self.s_pred]

if __name__ == "__main__":
    num_classes = 10
    IMAGE_SIZE = 416

    model = YOLOv3(num_class=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)

    print(out[0].shape)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes+5)
    print("Success")