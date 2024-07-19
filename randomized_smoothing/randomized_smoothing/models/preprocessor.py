# system modules
import os, sys, argparse
# sys.path.insert(0, "./utils")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from .utils import *
# from utils import *

class Preprocessor(nn.Module):
    def __init__(self):
        super(Preprocessor, self).__init__()
        my_act = GroupSort()
        self.conv1 = nn.Sequential(
            OrthoConv2d(in_channels=3, out_channels=6, kernel_size=3, init = "permutation"),
            my_act,
        )

        self.conv2 = nn.Sequential(
            OrthoConv2d(in_channels=6, out_channels=16, kernel_size=3, init = "permutation"),
            my_act,
        )

        self.conv3 = nn.Sequential(
            OrthoConv2d(in_channels=16, out_channels=32, kernel_size=3, init = "permutation"),
            my_act,
        )

        self.conv4 = nn.Sequential(
            OrthoConv2d(in_channels=32, out_channels=64, kernel_size=3, init = "permutation"),
            my_act,
        )


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        outputs = self.conv4(x)

        return outputs

if __name__ == '__main__':
    model = Preprocessor()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.size())