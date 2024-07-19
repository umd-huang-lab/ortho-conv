# system modules
import os, sys, argparse
# sys.path.insert(0, "./utils")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from .utils import *
# from utils import * 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # print("block x shape: ", x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print("block output: ", out.shape)
        out = self.bn2(self.conv2(out))
        # print("block output: ", out.shape)
        out += self.shortcut(x)
        # print("block output after shortcut: ", out.shape)
        out = F.relu(out)
        return out


class OrthoResNet_partial(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(OrthoResNet_partial, self).__init__()
        self.my_act = GroupSort()
        self.in_planes = 128
        self.conv1 = OrthoConv2d(3, 64, kernel_size=3, stride=1, init = "permutation")
        self.conv2 = OrthoConv2d(64, 64, kernel_size=3, stride=1, init = "permutation")
        self.conv3 = OrthoConv2d(64, 128, kernel_size=6, stride=2, init = "permutation")
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # print("layer 5 done!")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.my_act(self.conv1(x))
        # print("conv out 0 shape: ", out.shape)
        out = self.my_act(self.conv2(out))
        # print("conv out 1 shape: ", out.shape)
        out = self.my_act(self.conv3(out))
        # print("conv out 2 shape: ", out.shape)
        out = self.layer3(out)
        # print("conv out 3 shape: ", out.shape)
        out = self.layer4(out)
        # print("conv out 4 shape: ", out.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print("out shape: ", out.shape)
        out = self.linear(out)

        probability = F.softmax(out, dim=1)
        predictions = torch.argmax(probability, dim=-1)

        return out, predictions

def OrthoResNet18_partial():
    print("We are using ResNet18!")
    return OrthoResNet_partial(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])

if __name__ == '__main__':
    # model = LeNet_orth_full()
    # model = LeNet_orth_partial()
    # net = ResNet18()
    # net = OrthoResNet18()
    net = OrthoResNet18_partial()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())