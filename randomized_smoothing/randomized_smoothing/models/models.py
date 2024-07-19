# system modules
import os, sys, argparse
# sys.path.insert(0, "./utils")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from .utils import *
# from utils import * 

class LeNet_orth_full(nn.Module):
    def __init__(self):
        super(LeNet_orth_full, self).__init__()
        my_act = GroupSort()
        self.conv1 = nn.Sequential(
            OrthoConv2d(in_channels=1, out_channels=6, kernel_size=4),
            my_act,
        )

        self.conv2 = nn.Sequential(
            OrthoConv2d(in_channels=6, out_channels=16, kernel_size=4),
            my_act,
        )

        self.fc1 = nn.Sequential(
            OrthoLinear(in_features=16 * 7 * 7, out_features=84),
            my_act,
        )

        self.fc2 = nn.Sequential(
            OrthoLinear(in_features=84, out_features=84),
            my_act,
        )

        self.fc3 = nn.Sequential(
            OrthoLinear(in_features=84, out_features=10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        bottleneck = x.view(batch_size, -1)
        x = self.fc1(bottleneck)
        x = self.fc2(x)
        outputs = self.fc3(x)

        probability = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probability, dim=-1)

        return outputs, predictions

class LeNet_orth_partial(nn.Module):
    def __init__(self):
        super(LeNet_orth_partial, self).__init__()
        my_act = GroupSort()
        self.conv1 = nn.Sequential(
            OrthoConv2d(in_channels=1, out_channels=6, kernel_size=4),
            my_act,
        )

        self.conv2 = nn.Sequential(
            OrthoConv2d(in_channels=6, out_channels=16, kernel_size=4),
            my_act,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=16 * 7 * 7, out_features=84),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=84, out_features=84),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10),
        )


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        # print("x.shape: ", x.shape)
        x = self.conv2(x)
        # print("x.shape: ", x.shape)
        bottleneck = x.view(batch_size, -1)
        # print("bottoleneck shape: ", bottleneck.shape)
        x = self.fc1(bottleneck)
        # print("x.shape: ", x.shape)
        x = self.fc2(x)
        # print("x.shape: ", x.shape)
        outputs = self.fc3(x)

        probability = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probability, dim=-1)

        return outputs, predictions

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=16 * 4 * 4, out_features=84),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=84, out_features=84),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        # print("x.shape: ", x.shape)
        x = self.conv2(x)
        # print("x.shape: ", x.shape)
        bottleneck = x.view(batch_size, -1)
        # print("bottleneck.shape: ", bottleneck.shape)
        x = self.fc1(bottleneck)
        # print("x.shape: ", x.shape)
        x = self.fc2(x)
        # print("x.shape: ", x.shape)
        outputs = self.fc3(x)

        probability = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probability, dim=-1)

        return outputs, predictions

class OrthoConv_partial(nn.Module):
    def __init__(self, num_classes=10):
        super(OrthoConv_partial, self).__init__()
        self.my_act = GroupSort()
        self.conv1 = OrthoConv2d(3, 64, kernel_size=6, stride=2, init = "permutation")
        self.conv2 = OrthoConv2d(64, 64, kernel_size=6, stride=2, init = "permutation")
        self.conv3 = OrthoConv2d(64, 128, kernel_size=6, stride=2, init = "permutation")
        # self.conv4 = OrthoConv2d(128, 256, kernel_size=2, stride=1, init = "permutation")
        # self.conv5 = OrthoConv2d(256, 512, kernel_size=2, stride=1, init = "permutation")
        self.conv4 = OrthoConv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = OrthoConv2d(256, 512, kernel_size=2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.my_act(self.conv1(x))
        # print("conv out 0 shape: ", out.shape)
        out = self.my_act(self.conv2(out))
        # print("conv out 1 shape: ", out.shape)
        out = self.my_act(self.conv3(out))
        # print("conv out 2 shape: ", out.shape)
        out = self.my_act(self.conv4(out))
        # print("conv out 3 shape: ", out.shape)
        out = self.my_act(self.conv5(out))
        # print("conv out 4 shape: ", out.shape)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print("out shape: ", out.shape)
        out = self.linear(out)

        probability = F.softmax(out, dim=1)
        predictions = torch.argmax(probability, dim=-1)

        return out, predictions


if __name__ == '__main__':
    # model = LeNet_orth_full()
    model = OrthoConv_partial()
    y = model(torch.randn(1, 3, 32, 32))
    print(y.size())