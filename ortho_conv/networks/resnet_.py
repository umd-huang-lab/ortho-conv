import sys
import functools

import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Any, List

from modules import *


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        downsampling: str,
        conv_kernel:  int, 
        stride: int = 1,
        conv: Callable = Paraunitary,
        norm: Callable = None,
        actv: Callable = GroupSort,
    ) -> None:
        """
        Construction of a residual block for ResNet.

        """
        super(ResidualBlock, self).__init__()

        if stride > 1:
            if downsampling == "pool":
                # shortcut branch
                if norm is None:
                    self.shortcut = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel),
                        nn.AvgPool2d(stride, divisor_override = stride)
                    )
                else: # if norm is not None:
                    self.shortcut = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel),
                        nn.AvgPool2d(stride, divisor_override = stride),
                        norm(out_channels),
                    )

                # residual branch
                if norm is None: 
                    self.residual = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel),
                        nn.AvgPool2d(stride, divisor_override = stride),
                        actv(),
                        conv(out_channels, out_channels, conv_kernel)
                    )
                else: # if norm is not None:
                    self.residual = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel),
                        nn.AvgPool2d(stride, divisor_override = stride),
                        norm(out_channels), actv(),
                        conv(out_channels, out_channels, conv_kernel),
                        norm(out_channels)
                    )

            elif downsampling == "stride_wide":
                # shortcut branch
                if norm is None:
                    self.shortcut = conv(in_channels, out_channels, conv_kernel * stride, stride = stride)
                else: # if norm is not None:
                    self.shortcut = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel * stride, stride = stride),
                        norm(out_channels)
                    )

                # residual branch
                if norm is None:
                    self.residual = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel * stride, stride = stride),
                        actv(),
                        conv(out_channels, out_channels, conv_kernel)
                    )
                else: # if norm is not None:
                    self.residual = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel * stride, stride = stride),
                        norm(out_channels), actv(),
                        conv(out_channels, out_channels, conv_kernel),
                        norm(out_channels)
                    )

            elif downsampling == "stride_slim":
                # shortcut branch
                if norm is None:
                    self.shortcut = conv(in_channels, out_channels, stride, stride = stride)
                else: # if norm is not None:
                    self.shortcut = nn.Sequential(
                        conv(in_channels, out_channels, stride, stride = stride),
                        norm(out_channels)
                    )

                # residual branch
                if norm is None:
                    self.residual = nn.Sequential(
                        conv(in_channels, out_channels, stride, stride = stride),
                        actv(),
                        conv(out_channels, out_channels, conv_kernel)
                    )
                else: # if norm is not None
                    self.residual = nn.Sequential(
                        conv(in_channels, out_channels, stride, stride = stride),
                        norm(out_channels), actv(),
                        conv(out_channels, out_channels, conv_kernel),
                        norm(out_channels)
                    )

            else: # if downsampling not in ["pool", "stride_wide", "stride_slim"]:
                raise ValueError("The downsampling type is not supported.")

        else: # if stride == 1:
            # shortcut branch
            if in_channels == out_channels:
                self.shortcut = lambda inputs: inputs
            else: # if in_channels != out_channels:
                if norm is None:
                    self.shortcut = conv(in_channels, out_channels, conv_kernel)
                else: # if norm is not None:
                    self.shortcut = nn.Sequential(
                        conv(in_channels, out_channels, conv_kernel),
                        norm(out_channels)
                    )

            # residual branch
            if norm is None:
                self.residual = nn.Sequential(
                    conv( in_channels, out_channels, conv_kernel),
                    actv(),
                    conv(out_channels, out_channels, conv_kernel)
                )
            else: # if norm is not None:
                self.residual = nn.Sequential(
                    conv( in_channels, out_channels, conv_kernel),
                    norm(out_channels), actv(),
                    conv(out_channels, out_channels, conv_kernel),
                    norm(out_channels)
                )

        # skip connection
        self.skip_connection = ConvexCombo()
        self.skip_activation = actv()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the residual block.

        """

        # process 
        shortcut = self.shortcut(inputs)
        residual = self.residual(inputs)

        # merge
        outputs = self.skip_connection(residual, shortcut)
        outputs = self.skip_activation(outputs)

        return outputs


class _ResNet(nn.Module):

    def __init__(self,
        stages_repeats:  List[int],
        stages_strides:  List[int],
        stages_channels: List[int],
        downsampling: str = "pool",
        conv_kernel: int = 3,
        conv_dilate: int = 1,
        pool_kernel: int = 1,
        out_classes: int = 10,
        in_channels: int = 3,
        in_height: int = 32,
        in_width:  int = 32,
        linear: Callable = CayleyLinear,
        conv: Callable = Paraunitary,
        norm: Callable = None,
        actv: Callable = GroupSort,
        init: str = "permutation"
    ) -> None:
        """
        Construction of a ResNet.

        Arguments:
        ----------
        [network architecture]
        stages_repeats: int
            The number of blocks at each stage.
        stages_strides: int
            The stride for each stage.
        stages_channels: int
            The number of channels for each stage.

        downsampling: str
            The type of downsampling used in the residual block.
            Default: "pool"
        conv_kernel: int
            The kernel size for the convolutional layers.
            Default: 3
        conv_dilate: int
            The dilation for the convolutional layers.
            Default: 1
        
        pool_kernel: int
            The kernel size of the average pooling layer before fully-connected layers.
            Default: 1, i.e., no pooling layer.

        [input/output interfaces]
        out_classes: int
            The number of output classes.
            Default: 10
        in_channels: int
            The number of input channels.
            Default: 3
        in_height: int
            The height of the input images.
            Default: 32
        in_width: int
            The width of the input images.
            Default: 32

        [input/output interfaces]
        out_classes: int
            The number of output classes.
            Default: 10
        in_channels: int
            The number of input channels.
            Default: 3
        in_height: int
            The height of the input images.
            Default: 32
        in_width: int
            The width of the input images.
            Default: 32

        [orthogonal layers]
        linear: Callable
            THe fully-connected layer used in the ResNet.
            Default: CayleyLinear
        conv: Callable
            The convolutional layers used for the ShuffleNet.
            Default: Paraunitary
        init: str
            The initialization method for the convolutional layer.
            Default: "permuatation"

        """
        super(_ResNet, self).__init__()

        num_stages = len(stages_repeats)

        if len(stages_strides) != num_stages:
            raise ValueError('The stages_repeats is expected as a list of num_stages ints.')

        if len(stages_channels) != num_stages + 1:
            raise ValueError('The stages_channels is expected as a list of (num_stages + 1) ints.')

        if conv is Paraunitary:
            conv = functools.partial(conv, dilation = conv_dilate, init = init)

        # input convolutional layer
        out_channels = stages_channels.pop(0)

        if norm is None:
            self.conv = nn.Sequential(
                conv(in_channels, out_channels, conv_kernel),
                actv(),
            )
        else: # if norm is not None:
            self.conv = nn.Sequential(
                conv(in_channels, out_channels, conv_kernel),
                norm(out_channels), actv()
            )

        in_channels = out_channels

        # residual blocks
        self.blocks = nn.ModuleList()

        for s in range(num_stages):
            out_channels = stages_channels[s]
            stride = stages_strides[s]

            self.blocks.append(ResidualBlock(
                in_channels, out_channels, downsampling, conv_kernel, stride, conv, norm, actv))

            for l in range(stages_repeats[s] - 1):
                self.blocks.append(ResidualBlock(
                    out_channels, out_channels, downsampling, conv_kernel, 1, conv, norm, actv))

            in_height //= stride
            in_width  //= stride
            in_channels = out_channels

        # (optional) pooling layer
        if pool_kernel > 1:
            self.pool = nn.Sequential(
                nn.AvgPool2d(pool_kernel, divisor_override = pool_kernel),
                nn.Flatten()
            ) 
        else: # if pool_kernel == 1:
            self.pool = nn.Flatten()
        
        in_height //= pool_kernel
        in_width  //= pool_kernel
        in_channels *= in_height * in_width

        # fully-connected layers
        out_channels = in_channels // 4

        self.full = nn.Sequential(
            linear(in_channels, out_channels),
            actv(),
            linear(out_channels, out_classes)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the ResNet.

        Argument:
        ---------
        inputs: a 4th-order Tensor of size 
            [batch_size, in_channels, in_height, in_width].
            The inputs to the ResNet.

        Return:
        -------
        outputs: a 2nd-order Tensor of size 
            [batch_size, out_classes].
            The outputs of the ResNet.

        """

        # input convolutional layer
        inputs = self.conv(inputs)

        # residual blocks
        for block in self.blocks:
            inputs = block(inputs)

        # (optional) pooling layer
        outputs = self.pool(inputs)

        # output fully-connected layer
        outputs = self.full(outputs)

        return outputs


def ResNet(
    num_layers: int,
    base_channels: int = 64,
    downsampling: str = "pool",
    conv_kernel: int = 3,
    conv_dilate: int = 1,
    pool_kernel: int = 2,
    out_classes: int = 10,
    in_channels: int = 3,
    in_height: int = 32,
    in_width:  int = 32,
    linear: Callable = CayleyLinear,
    conv: Callable = Paraunitary,
    norm: Callable = None,
    actv: Callable = GroupSort,
    init: str = "permutation"
) -> _ResNet:
    
    if num_layers % 6 == 2:
        stages_repeats = [num_layers // 6] * 3
    else: # if num_layers not in [8, 20, 32, 44, 56]:
        raise ValueError("The number of layers is not supported.")

    num_stages = len(stages_repeats)

    stages_strides  = [2] * num_stages
    stages_channels = [base_channels] * (num_stages + 1)
    for s in range(num_stages + 1):
        stages_channels[s] *= 2 ** s

    return _ResNet(
        stages_repeats  = stages_repeats,
        stages_strides  = stages_strides,
        stages_channels = stages_channels,
        downsampling = downsampling,
        conv_kernel = conv_kernel,
        conv_dilate = conv_dilate,
        pool_kernel = pool_kernel,
        out_classes = out_classes,
        in_channels = in_channels,
        in_height   = in_height,
        in_width    = in_width,
        linear = linear,
        conv = conv,
        norm = norm,
        actv = actv,
        init = init,
    )


def WideResNet(
    num_layers: int,
    base_channels: int = 16,
    widen_factor: int = 4,
    downsampling: str = "pool",
    conv_kernel: int = 3,
    conv_dilate: int = 1,
    pool_kernel: int = 4,
    out_classes: int = 10,
    in_channels: int = 3,
    in_height: int = 32,
    in_width:  int = 32,
    linear: Callable = CayleyLinear,
    conv: Callable = Paraunitary,
    norm: Callable = None,
    actv: Callable = GroupSort,
    init: str = "permutation"
) -> _ResNet:

    if num_layers % 6 == 4:
        stages_repeats = [num_layers // 6] * 3
    else: # if num_layers not in [10, 22, 34, 46]:
        raise ValueError("The number of layers is not supported.")

    num_stages = len(stages_repeats)

    stages_strides  = [1] + [2] * (num_stages - 1)
    stages_channels = [base_channels] * (num_stages + 1)
    for s in range(num_stages):
        stages_channels[s + 1] *= 2 ** s * widen_factor

    return _ResNet(
        stages_repeats  = stages_repeats,
        stages_strides  = stages_strides,
        stages_channels = stages_channels,
        downsampling = downsampling,
        conv_kernel = conv_kernel,
        conv_dilate = conv_dilate,
        pool_kernel = pool_kernel,
        out_classes = out_classes,
        in_channels = in_channels,
        in_height   = in_height,
        in_width    = in_width,
        linear = linear,
        conv = conv,
        norm = norm,
        actv = actv,
        init = init,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # batch size and input dimensions
    batch_size, in_channels, in_height, in_width, out_classes = 256, 3, 32, 32, 10

    # hyperparameters for convolution
    conv_kernel, conv_dilate, init = 3, 1, "permutation"

    # hyperparameters for ResNet
    for num_layers in [8, 20]:
        for downsampling in ["pool", "stride_wide", "stride_slim"]:
            print("Testing ResNet%d_%s" % (num_layers, downsampling))

            for linear, conv, norm, actv in [
                ['nn.Linear', 'PlainConv', 'nn.BatchNorm2d', 'nn.ReLU'],
                ['CayleyLinear', 'Paraunitary', None, 'GroupSort']
                
            ]:
                print("conv: %s, linear: %s, norm: %s, actv: %s" % (linear, conv, norm, actv))

                linear = eval(linear)
                conv = eval(conv)
                norm = eval(norm) if norm is not None else None
                actv = eval(actv)

                # initialize the ResNet
                network = ResNet(
                    num_layers  = num_layers,
                    downsampling = downsampling,
                    conv_kernel = conv_kernel,
                    conv_dilate = conv_dilate,
                    out_classes = out_classes,
                    in_channels = in_channels,
                    in_height   = in_height,
                    in_width    = in_width,
                    linear = linear,
                    conv   = conv,
                    norm   = norm,
                    actv   = actv,
                    init   = init,
                ).to(device)

                # evaluate the network using randomized input
                inputs  = torch.randn(batch_size, in_channels,
                    in_height, in_width, requires_grad = True).to(device)

                outputs = network(inputs)
                print(outputs.size())

    # hyperparameters for WideResNet
    for num_layers, widen_factor in [[10, 4], [22, 1]]:
        for downsampling in ["pool", "stride_wide", "stride_slim"]:
            print("Testing WideResNet%d-%d_%s" % (num_layers, widen_factor, downsampling))

            for linear, conv, norm, actv in [
                ['nn.Linear', 'PlainConv', 'nn.BatchNorm2d', 'nn.ReLU'],
                ['CayleyLinear', 'Paraunitary', None, 'GroupSort']
                
            ]:
                print("conv: %s, linear: %s, norm: %s, actv: %s" % (linear, conv, norm, actv))

                linear = eval(linear)
                conv = eval(conv)
                norm = eval(norm) if norm is not None else None
                actv = eval(actv)

                # initialize the WideResNet
                network = WideResNet(
                    num_layers = num_layers,
                    downsampling = downsampling,
                    conv_kernel = conv_kernel,
                    conv_dilate = conv_dilate,
                    out_classes = out_classes,
                    in_channels = in_channels,
                    in_height = in_height,
                    in_width  = in_width,
                    linear = linear,
                    conv   = conv,
                    norm   = norm,
                    actv   = actv,
                    init   = init,
                ).to(device)

                # evaluate the network using randomized input
                inputs  = torch.randn(batch_size, in_channels,
                    in_height, in_width, requires_grad = True).to(device)

                outputs = network(inputs)
                print(outputs.size())
