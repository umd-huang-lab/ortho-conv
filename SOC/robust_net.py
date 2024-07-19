import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class MinMax(nn.Module):
    def __init__(self):
        super(MinMax, self).__init__()

    def forward(self, z, axis=1):
        a, b = z.split(z.shape[axis] // 2, axis)
        c, d = torch.min(a, b), torch.max(a, b)
        return torch.cat([c, d], dim=axis)
    

class LipBlock(nn.Module):
    def __init__(self, in_planes, planes, conv_module, stride = 1, kernel_size = 3):
        super(LipBlock, self).__init__()

        self.activation = MinMax()

        padding = kernel_size // 2
        kernel_size = kernel_size * stride

        self.conv = conv_module(in_planes, planes * stride, 
            kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, x):
        x = self.activation(self.conv(x))
        return x


class _LipNet(nn.Module):
    def __init__(self, block, num_blocks, conv, linear, kernel_size = 3,
        in_shape = 32, in_channels = 3, num_classes = 10, base_channels = 32,
    ):
        super(_LipNet, self).__init__()

        self.in_shape  = in_shape
        self.in_planes = in_channels

        self.layer1 = self._make_layer(block,  base_channels, num_blocks[0], conv, stride = 2, kernel_size = kernel_size)
        self.layer2 = self._make_layer(block, self.in_planes, num_blocks[1], conv, stride = 2, kernel_size = kernel_size)
        self.layer3 = self._make_layer(block, self.in_planes, num_blocks[2], conv, stride = 2, kernel_size = kernel_size)
        self.layer4 = self._make_layer(block, self.in_planes, num_blocks[3], conv, stride = 2, kernel_size = kernel_size)
        self.layer5 = self._make_layer(block, self.in_planes, num_blocks[4], conv, stride = 2, kernel_size = kernel_size)

        self.linear = linear(self.in_shape**2 * self.in_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, conv_module, stride, kernel_size):
        strides = [1] * (num_blocks - 1) + [stride]
        kernel_sizes = [kernel_size] * num_blocks

        layers = []
        for (stride, kernel_size) in zip(strides, kernel_sizes):
            if self.in_shape % stride == 0:
                self.in_shape //= stride
            else: # if self.in_shape % stride != 0:
                stride = 1

            if kernel_size % 2 == 0:
                kernel_size = min(kernel_size, self.in_shape)
            else: # if kernel_size % 2 == 1:
                if self.in_shape % 2 == 0:
                    kernel_size = min(kernel_size, self.in_shape + 1)
                else: # if self.in_shape % 2 == 1:
                    kernel_size = min(kernel_size, self.in_shape)

            print("in_shape: %d; kernel_size: %d" % (self.in_shape, kernel_size))

            layers.append(block(self.in_planes, planes, conv_module, stride, kernel_size))
            self.in_planes = planes * stride

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


def LipNet(
    conv, linear,
    kernel_size = 3,
    in_shape = 32,
    in_channels = 3,
    base_channels = 32,
    num_layers = 20,
    num_classes = 10
):
    if num_layers % 5 == 0:
        num_blocks_list = [num_layers // 5] * 5
    else: # if num_layers % 5 != 0:
        raise ValueError('The number of layers is not supported.')

    return _LipNet(LipBlock, num_blocks_list, conv, linear, kernel_size, 
                   in_shape, in_channels, num_classes, base_channels)
