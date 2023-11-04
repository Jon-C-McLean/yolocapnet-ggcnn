import torch
import torch.nn as nn
from utils.config import parse_config

from yolocapnet.layers.conv import Conv, pad
from yolocapnet.layers.sampling import Upsample
from yolocapnet.layers.misc import EmptyLayer

# Inspired by https://github.com/ayooshkathuria/pytorch-yolo-v3/

def build_modules(blocks):
    network_info = blocks[0] # This should really be checked to ensure the first block is actually the network info

    modules = nn.ModuleList()
    index = 0
    prev_filters = 3

    output_filters = []

    for x in blocks:
        module = nn.Sequential()

        if x['type'] == 'net':
            continue

        if x['type'] == 'convolutional':
            batch_normalize = int(x['batch_normalize'])
            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            padding = int(x['pad'])
            padding = pad(kernel_size, stride)
            activation = x['activation']

            bias = not batch_normalize

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
            module.add("conv_{0}".format(index), conv)

            if batch_normalize:
                module.add_module("batch_norm_{0}".format(index), nn.BatchNorm2d(filters))
            
            if activation == 'leaky':
                module.add_module("leaky_{0}".format(index), nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'linear':
                module.add_module("linear_{0}".format(index), nn.Linear(filters, filters)) # Is this correct?
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            module.add_module('upsample_{0}'.format(index), Upsample(stride))
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')

            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            
            if start >0:
                start = start - index
            
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        elif x['type'] == 'yolo':
            pass

        modules.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1

    return network_info, modules

def build_modules_from_config(file):
    blocks = parse_config(file)
    return build_modules(blocks)