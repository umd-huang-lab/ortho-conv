import sys
import functools

import torch
import torch.nn as nn

from torch import Tensor
from typing import Callable, Any, List

from modules import *


def Conv_K(
    in_channels:  int,
    out_channels: int,
    kernel_size:  int,
    conv: Callable = CayleyConv
) -> nn.Module:
    """
    Construction of a standard convolutional module.

    """
    if kernel_size % 2 == 1:
        padding = kernel_size // 2
    else: # if kernel_size % 2 == 0:
        raise ValueError("The kernel_size is expected as an odd int.")

    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size),
        GroupSort()
    )


def Conv_S(
    in_channels:  int,
    out_channels: int,
    kernel_size:  int,
    downsampling: str,
    stride: int,
    conv: Callable = CayleyConv
) -> nn.Module:
    """
    Construction of a strided convolutional module.

    """
    if downsampling == "pool":
        return nn.Sequential(
            conv(in_channels, out_channels, kernel_size),
            nn.AvgPool2d(stride, divisor_override = stride),
            GroupSort()
        )

    elif downsampling == "stride_wide":
        return nn.Sequential(
            conv(in_channels, out_channels, kernel_size * stride, stride = stride),
            GroupSort()
        )

    elif downsampling == "stride_slim":
        return nn.Sequential(
            conv(in_channels, out_channels, stride, stride = stride),
            GroupSort()
        )

    else: # if divisor_override not in ["pool", "stride_wide", "stride_slim"]:
        raise ValueError("The downsampling type is not supported.")


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        downsampling: str,
        kernel_size: int,
        stride: int = 1,
        conv: Callable = CayleyConv
    ) -> None:
        """
        Construction of a building block for ConvNet.

        """
        super(BasicBlock, self).__init__()

        if stride > 1:
            self.block = nn.Sequential(
                Conv_S( in_channels, out_channels, kernel_size, downsampling, stride, conv),
                Conv_K(out_channels, out_channels, kernel_size, conv)
            )
        else: # if stride == 1:
            self.block = nn.Sequential(
                Conv_K( in_channels, out_channels, kernel_size, conv),
                Conv_K(out_channels, out_channels, kernel_size, conv)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the building block.

        """

        outputs = self.block(inputs)

        return outputs


class _ConvNet(nn.Module):

    def __init__(
        self,
        stages_repeats:  List[int],
        stages_strides:  List[int],
        stages_channels: List[int],
        downsampling: str = "pool",
        conv_kernel: int = 3,
        conv_dilate: int = 1,
        pool_kernel: int = 2,
        out_classes: int = 10,
        in_channels: int = 3,
        in_height: int = 32,
        in_width:  int = 32,
        linear: Callable = CayleyLinear,
        conv: Callable = CayleyConv,
        init: str = "permutation"
    ) -> None:
        """
        Construction of a ShuffleNet.

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
            The type of downsampling used in the shuffling block.
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

        [orthogonal layers]
        linear: Callable
            THe fully-connected layer used in the ShuffleNet.
            Default: CayleyLinear
        conv: Callable
            The convolutional layers used for the ShuffleNet.
            Default: CayleyConv
        init: str
            The initialization method for the convolutional layer.
            Default: "permuatation"

        """
        super(_ConvNet, self).__init__()

        num_stages = len(stages_repeats)

        if len(stages_strides) != num_stages:
            raise ValueError('The stages_repeats is expected as a list of num_stages ints.')

        if len(stages_channels) != num_stages + 1:
            raise ValueError('The stages_channels is expected as a list of (num_stages + 1) ints.')

        if conv is Paraunitary:
            conv = functools.partial(conv, dilation = conv_dilate, init = init)

        if conv is BjorckConv:
            conv = functools.partial(conv, bjorck_thres = thres, init = init)

        if linear is BjorckLinear:
            linear = functools.partial(linear, bjorck_thres = thres, init = init)

        # input convolutional layer
        out_channels = stages_channels.pop(0)

        self.conv = Conv_K(in_channels, out_channels, conv_kernel, conv)

        in_channels = out_channels

        # shuffling blocks
        self.blocks = nn.ModuleList()

        for s in range(num_stages):
            out_channels = stages_channels[s]
            stride = stages_strides[s]

            self.blocks.append(BasicBlock(in_channels, out_channels, downsampling, conv_kernel, stride, conv))

            for l in range(stages_repeats[s] - 1):
                self.blocks.append(BasicBlock(out_channels, out_channels, downsampling, conv_kernel, 1, conv))

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
            GroupSort(),
            linear(out_channels, out_classes)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the ShuffleNet.

        Argument:
        ---------
        inputs: a 4th-order Tensor of size 
            [batch_size, in_channels, in_height, in_width].
            The inputs to the ShuffleNet.

        Return:
        -------
        outputs: a 2nd-order Tensor of size 
            [batch_size, out_classes].
            The outputs of the ShuffleNet.

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


def ConvNet(
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
    conv: Callable = CayleyConv,
    init: str = "permuatation"
) -> _ConvNet:
    
    if num_layers % 6 == 2:
        stages_repeats = [num_layers // 6] * 3
    else: # if num_layers not in [8, 20, 32, 44, 56]:
        raise ValueError("The number of layers is not supported.")

    num_stages = len(stages_repeats)

    stages_strides  = [2] * num_stages
    stages_channels = [base_channels] * (num_stages + 1)
    for s in range(num_stages + 1):
        stages_channels[s] *= 2 ** s

    return _ConvNet(
        stages_repeats  = stages_repeats,
        stages_strides  = stages_strides,
        stages_channels = stages_channels,
        downsampling = downsampling,
        conv_kernel = conv_kernel,
        pool_kernel = pool_kernel,
        out_classes = out_classes,
        in_channels = in_channels,
        in_height   = in_height,
        in_width    = in_width,
        linear = linear,
        conv = conv,
        init = init
    )


def WideConvNet(
    num_layers: int,
    base_channels: int = 16,
    widen_factor: int = 8,
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
    init: str = "permutation"
) -> _ConvNet:

    if num_layers % 6 == 4:
        stages_repeats = [num_layers // 6] * 3
    else: # if num_layers not in [10, 22, 34, 46]:
        raise ValueError("The number of layers is not supported.")

    num_stages = len(stages_repeats)

    stages_strides  = [1] + [2] * (num_stages - 1)
    stages_channels = [base_channels] * (num_stages + 1)
    for s in range(num_stages):
        stages_channels[s + 1] *= 2 ** s * widen_factor

    return _ConvNet(
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
        init = init
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # batch size and input dimensions
    batch_size, in_channels, in_height, in_width, out_classes = 256, 3, 32, 32, 10

    # hyperparameters for convolution
    conv_kernel, conv_dilate, init = 3, 1, "permutation"

    # hyperparameters for the ConvNet
    for num_layers in [8, 20]:
        for downsampling in ["pool", "stride_wide", "stride_slim"]:
            print("Testing ConvNet%d_%s" % (num_layers, downsampling))

            for conv, linear in [[Paraunitary, CayleyLinear], [CayleyConv, CayleyLinear]]:
                print("conv: %s, linear: %s" % (conv.__name__, linear.__name__))

                # initialize the ShuffleNet
                network = ConvNet(
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
                    init   = init,
                ).to(device)

                # evaluate the network using randomized input
                inputs  = torch.randn(batch_size, in_channels, 
                    in_height, in_width, requires_grad = True).to(device)

                outputs = network(inputs)
                print(outputs.size())

    # hyperparamters for WideConvNet
    for num_layers, widen_factor in [[10, 4], [22, 2]]:
        for downsampling in ["pool", "stride_wide", "stride_slim"]:
            print("Testing WideConvNet%d-%d_%s" % (num_layers, widen_factor, downsampling))

            for conv, linear in [[Paraunitary, CayleyLinear], [Paraunitary, CayleyLinear]]:
                print("conv: %s, linear: %s" % (conv.__name__, linear.__name__))

                # initialize the WideConvNet
                network = WideConvNet(
                    num_layers = num_layers,
                    widen_factor = widen_factor,
                    downsampling = downsampling,
                    conv_kernel = conv_kernel,
                    conv_dilate = conv_dilate,
                    out_classes = out_classes,
                    in_channels = in_channels,
                    in_height = in_height,
                    in_width  = in_width,
                    linear = linear,
                    conv   = conv,
                    init   = init,
                ).to(device)

                # evaluate the network using randomized input
                inputs  = torch.randn(batch_size, in_channels,
                    in_height, in_width, requires_grad = True).to(device)

                outputs = network(inputs)
                print(outputs.size())
