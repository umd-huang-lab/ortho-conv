import argparse
import torch
import numpy as np

from models import *

import sys
sys.path.append('../layers')
from OrthoConv2d import OrthoConv2d
from OrthoConvTranspose2d import OrthoConvTranspose2d


parser = argparse.ArgumentParser()

parser.add_argument('--runs',  default = 100, type = int)
parser.add_argument('--batch', default = 256, type = int)

parser.add_argument('--width', default = 16, type = int)
parser.add_argument('--height', default = 16, type = int)
parser.add_argument('--channels', default = 64, type = int)
parser.add_argument('--kernel_size', default = 3, type = int)

args = parser.parse_args()

# Experiment 1: standard convolutional layer for various methods
_convs = ["Paraunitary", "CayleyConv", "BCOP", "SVCM", "RKO", "OSSN"]
ratios = {}

for _conv in _convs:
    conv = eval(_conv)
    print("conv: %s" % _conv)

    # use uniform init for Paraunitary
    if conv is Paraunitary:
        conv = functools.partial(conv, init = "uniform")

    ratio = np.zeros(args.batch * args.runs)

    for run in range(args.runs):
        # initialize the layer
        layer = conv(args.channels, args.channels, args.kernel_size).cuda()

        # SCVM projection
        if conv is SVCM:
            inputs  = torch.randn(args.batch, args.channels, args.height, args.width).cuda()
            outputs = layer(inputs)

            for m in layer.modules():
                if hasattr(m, '_project'):
                    m._project()

        # evaluate the layer using randomized inputs
        inputs  = torch.randn(args.batch, args.channels, args.height, args.width).cuda()
        outputs = layer(inputs)

        inputs  = torch.reshape(inputs,  (args.batch, -1))
        outputs = torch.reshape(outputs, (args.batch, -1))

        # check norm perservation
        ratio_ = torch.norm(outputs, dim = 1) / torch.norm(inputs, dim = 1)
        ratio[run * args.batch: (run + 1) * args.batch] = ratio_.data.cpu().numpy()

    ratios[_conv] = ratio
    print("mean: %s, std: %s" % (np.mean(ratio), np.std(ratio)))


# Experiment 2: our approach for variants of convolutional layers

for groups in [1, 4, 16]:

    # dilated convolutional layer
    for dilation in [2, 4, 8]:
        print("groups = %d, dilation = %d" % (groups, dilation))

        ratio = np.zeros(args.batch * args.runs)

        for run in range(args.runs):
            # initialize the layer
            layer = OrthoConv2d(args.channels, args.channels, args.kernel_size, 
                dilation = dilation, groups = groups, init = "uniform").cuda()

            # evaluate the layer using randomized inputs
            inputs  = torch.randn(args.batch, args.channels, args.height, args.width).cuda()
            outputs = layer(inputs)

            inputs  = torch.reshape(inputs,  (args.batch, -1))
            outputs = torch.reshape(outputs, (args.batch, -1))

            ratio_ = torch.norm(outputs, dim = 1) / torch.norm(inputs, dim = 1)
            ratio[run * args.batch: (run + 1) * args.batch] = ratio_.data.cpu().numpy()

        print("mean: %s, std: %s" % (np.mean(ratio), np.std(ratio)))

    # strided convolutional layer
    for stride in [2, 4]:
        print("groups = %d, stride = %d" % (groups, stride))

        ratio = np.zeros(args.batch * args.runs)

        for run in range(args.runs):
            # initialize the layer
            layer = OrthoConv2d(args.channels, args.channels * stride ** 2, 
                args.kernel_size * stride, stride = stride, groups = groups, init = "uniform").cuda()

            # evaluate the layer using randomized inputs
            inputs  = torch.randn(args.batch, args.channels, args.height, args.width).cuda()
            outputs = layer(inputs)

            inputs  = torch.reshape(inputs,  (args.batch, -1))
            outputs = torch.reshape(outputs, (args.batch, -1))

            ratio_ = torch.norm(outputs, dim = 1) / torch.norm(inputs, dim = 1)
            ratio[run * args.batch: (run + 1) * args.batch] = ratio_.data.cpu().numpy()

        print("mean: %s, std: %s" % (np.mean(ratio), np.std(ratio)))

    # tranposed strided convolutional layer
    for stride in [2, 4]:
        print("groups = %d, stride = 1/%d" % (groups, stride))

        ratio = np.zeros(args.batch * args.runs)

        for run in range(args.runs):
            # initialize the layer
            layer = OrthoConvTranspose2d(args.channels, args.channels // stride ** 2, 
                args.kernel_size * stride, stride = stride, groups = groups, init = "uniform").cuda()

            # evaluate the layer using randomized inputs
            inputs  = torch.randn(args.batch, args.channels, args.height, args.width).cuda()
            outputs = layer(inputs)

            inputs  = torch.reshape(inputs,  (args.batch, -1))
            outputs = torch.reshape(outputs, (args.batch, -1))

            ratio_ = torch.norm(outputs, dim = 1) / torch.norm(inputs, dim = 1)
            ratio[run * args.batch: (run + 1) * args.batch] = ratio_.data.cpu().numpy()

        print("mean: %s, std: %s" % (np.mean(ratio), np.std(ratio)))
