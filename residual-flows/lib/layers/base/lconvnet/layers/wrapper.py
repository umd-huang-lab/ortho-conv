import torch.nn as nn
from lconvnet.layers.invertible_downsampling import PixelUnshuffle2d

class LipschitzConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        coeff=1.0,
        conv_module=None,
    ):
        """
        A wrapper for LipschitzConv2d. It uses invertible downsampling to mimic striding.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.coeff = coeff
        self.bias = bias

        # compute the in_channels based on stride
        # the invertible downsampling is applied before the convolution
        self.true_in_channels = in_channels * stride * stride
        self.true_out_channels = out_channels
        self.true_stride = 1
        assert kernel_size % stride == 0

        # compute the kernel size of the actual convolution based on stride
        self.true_kernel_size = kernel_size // stride
        self.shuffle = PixelUnshuffle2d(stride)
        self.conv = conv_module(
            self.true_in_channels,
            self.true_out_channels,
            kernel_size=self.true_kernel_size,
            stride=1,
            padding=self.true_kernel_size // 2,
            bias=self.bias
        )
        
    def forward(self, x):
        x = self.shuffle(x)
        x = self.conv(x) * self.coeff
        return x


class LipschitzLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        coeff=1.0,
        linear_module=None,
    ):
        """
        A wrapper for LipschitzLinear.

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.bias = bias

        self.linear = linear_module(
            self.in_features,
            self.out_features,
            self.bias
        )

    def forward(self, x):
        x = self.linear(x) * self.coeff
        return x
