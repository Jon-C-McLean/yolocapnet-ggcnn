import os, time
import argparse as ap

import torch
import torch.nn as nn

from yolocapnet.yolo import Darknet53, YOLOv3
from yolocapnet.layers.blocks import ResidualBlock

parser = ap.ArgumentParser(description='YoloCapNet-GGCNN')
# parser.add_argument('data', help="Dataset Path")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--learning-rate', default=0.01, type=float, help="Learning rate")
parser.add_argument('--batch-size', default=32, type=int, help="Batch size")

args = parser.parse_args()

device = torch.device('mps' if torch.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu')

def train(x_train, y_train, epoch, args):
    pass

darknet = Darknet53(block=ResidualBlock, num_class=10, init_weights=True)#.to(device)
yolo = YOLOv3(num_class=10)#.to(device)


if __name__ == '__main__':
    print("Using device: {}".format(device))

    from torchsummary import summary
    # summary(darknet, (3, 416, 416))
    summary(yolo, (3, 416, 416))
    

