import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .utils_bjorck import _circular_pad, _power_iteration_conv2d_circ, _bjorck_orthonormalize_conv2d_circ
from .utils_bjorck import _power_iteration, _bjorck_orthonormalize

from torch import Tensor
from typing import Callable


class BjorckLinear(nn.Module):

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias: bool = False,
        init: str = "permutation",
        power_iters: int = 10,
        power_thres: float = 1e-6,
        bjorck_order: int = 1,
        bjorck_iters: int = 20,
        bjorck_thres: float = 1e-6
    ) -> None:
        """
        Orthogonal linear layer with Björck orthonormalization.
        
        Arguments:
        ----------
        [hyper-parameters for linear layer]
        in_features: int
            The number of input features to the layer.
        out_features: int
            The number of output features of the layer.
        bias: bool
            Whether to add a learnable bias to the output.
            Default: False

        init: str ("identical", "reverse", or permutation")
            The initialization method of the orthogonal layer.
            Default: "permutation"
        
        [hyper-parameters for power iteration]
        power_iters: int
            Then maximum number for power iteration.
            Default: 10
        power_thres: float
            The absolute tolerance for power iteration.
            Default: 1e-6
        
        [hyper-parameters for Björck orthonormalization]
        bjorck_order: int
            The order of Taylor's expansion for Björck orthonormalization.
            Default: 1
        iters: int
            The maximum iterations for Björck orthonormalization.
            Default: 20
        thres: float
            The absolute tolerance for Björck orthonormalization.
            Default: 1e-6

        """
        super(BjorckLinear, self).__init__()

        ## [hyper-parameters for the Björck linear layer]
        self.in_features  =  in_features
        self.out_features = out_features

        if self.in_features > self.out_features:
            warnings.warn("The layer is made row-orthogonal.")
        elif self.in_features < self.out_features: 
            warnings.warn("The layer is made column-orthogonal.")

        self.power_iters = power_iters
        self.power_thres = power_thres

        self.bjorck_order = bjorck_order
        self.bjorck_iters = bjorck_iters
        self.bjorck_thres = bjorck_thres

        ## [initialization of the layer parameters]
        self.weight = None

        self.params = nn.Parameter(torch.zeros(self.out_features, self.in_features))
        self.v_vec = nn.Parameter(torch.randn(self.in_features), requires_grad = False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else: # if not bias:
            self.register_parameter("bias", None)

        self._initialize(init)

        ## [functionals for forward/inverse computation]
        self._forward = lambda inputs, weights, biases: F.linear(inputs, weights, biases)

        if bias:
            self._inverse = lambda inputs, weights, biases: F.linear(inputs - biases, weights.t())
        else: # if not bias:
            self._inverse = lambda inputs, weights, biases: F.linear(inputs, weights.t())

    def _initialize(self, init: str = "permutation") -> None:
        """
        Initialization of the Björck linear layer.

        Argument:
        ---------
        init: str ("identical", "reverse", or "permutation")
            The initialization method of the Björck linear layer.
            Default: "permutation"

        """
        if init not in ["identical", "reverse", "permutation"]:
            raise ValueError("The initialization method %s is not supported." % init)

        max_features = max(self.out_features, self.in_features)
        min_features = min(self.out_features, self.in_features)

        perm = torch.randperm(max_features)[:min_features]
        if init == "identical":
            perm, _ = torch.sort(perm, descending = False) 
        elif init == "reverse":
            perm, _ = torch.sort(perm, descending = True)

        matrix = torch.zeros(self.out_features, self.in_features)
        if self.out_features < self.in_features:
            matrix[:, perm] = torch.eye(min_features)
        else: # if self.out_features >= self.in_features:
            matrix[perm, :] = torch.eye(min_features)

        self.params.data = matrix

    def _project(self) -> None:
        """
        Björck orthonormalization of the weights.
        
        """
        with torch.no_grad():
            scale, self.v_vec.data = _power_iteration(self.params,
                self.power_iters, self.power_thres, self.v_vec)

        self.weight = _bjorck_orthonormalize(self.params / scale,
            self.bjorck_order, self.bjorck_iters, self.bjorck_thres)

    def forward(self, inputs: Tensor, inverse: bool = False, cache: bool = False) -> Tensor:
        """
        Computation of the BjorckLinear layer. 
        
        Arguments:
        ----------
        inputs: a matrix of size [batch_size, in_features]
            The input to the Björck linear layer.

        inverse: bool
            Whether to compute the inverse of the layer.
            Default: False

        cache: bool
            Whether to use the cached weights for computation.
            Default: False

        Return:
        -------
        outputs: a matrix of size [batch_size, out_features]
            The output of the Björck linear layer.

        """
        if self.weight is None or (self.training and not cache):
            self._project()

        if not inverse:
            outputs = self._forward(inputs, self.weight, self.bias)
        else: # if inverse:
            outputs = self._inverse(inputs, self.weight, self.bias)

        return outputs


class DilatedBjorckConv2d(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride: int = 1,
        bias: bool = False,
        padding: int = None,
        init: str = "permutation",
        power_iters: int = 10,
        power_thres: float = 1e-6,
        bjorck_order: int = 1,
        bjorck_iters: int = 20,
        bjorck_thres: float = 1e-6
    ) -> None:
        """
        Orthogonal dilated 2D-convolution layer with Björck orthonormalization.

        Arguments:
        ----------
        [hyper-parameters for 2D-convolution layer]
        in_channels: int
            The number of input channels to the dilated 2D-convolution layer.
        out_channels: int
            The number of output channels of the dilated 2D-convolution layer.
        kernel_size: int
            The length of the convolutional kernels.
        bias: bool
            Whether to add a learnable bias to the output.
            Default: False

        init: str ("identical", "reverse", "permutation")
            The initialization method of the orthogonal layer.
            Default: "permutation"

        [hyper-parameters for power iteration]
        power_iters: int
            Then maximum number for power iteration.
            Default: 10
        power_thres: float
            The absolute tolerance for power iteration.
            Default: 1e-6
        
        [hyper-parameters for Björck orthonormalization]
        bjorck_order: int
            The order of Taylor's expansion for Björck orthonormalization.
            Default: 1
        iters: int
            The maximum iterations for Björck orthonormalization.
            Default: 20
        thres: float
            The absolute tolerance for Björck orthonormalization.
            Default: 1e-6

        """
        super(DilatedBjorckConv2d, self).__init__()

        ## [hyper-parameters for the Björck 2D-convolution layer]
        self.in_channels  =  in_channels
        self.out_channels = out_channels

        if self.in_channels > self.out_channels:
            warnings.warn("The layer is made row-orthogonal.")
        if self.in_channels < self.out_channels:
            warnings.warn("The layer is made column-orthogonal.")

        self.kernel_size = _pair(kernel_size)

        self.power_iters = power_iters
        self.power_thres = power_thres

        self.bjorck_order = bjorck_order
        self.bjorck_iters = bjorck_iters
        self.bjorck_thres = bjorck_thres

        ## [initialization of the layer parameters]
        self.weight = None

        self.params = nn.Parameter(torch.zeros(
            self.out_channels, self.in_channels, *self.kernel_size))

        self.v_vec = nn.Parameter(torch.randn(1, 
            self.in_channels, *self.kernel_size), requires_grad = False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else: # if not bias:
            self.register_parameter('bias', None)

        self._initialize(init)

    def _initialize(self, init: str = "permutation") -> None:
        """
        Initialization of Björck 2D-convolution layer.

        Argument:
        ---------
        init: str ("identical", "reverse", or "permutation")
            The initialization method of the Björck 1D-convolution layer.
            Default: "permutation"

        """
        if init not in ["identical", "reverse", "permutation"]:
            raise ValueError("The initialization method %s is not supported." % init)

        max_channels = max(self.out_channels, self.in_channels)
        min_channels = min(self.out_channels, self.in_channels)

        perm = torch.randperm(max_channels)[:min_channels]
        if init == "identical":
            perm, _ = torch.sort(perm, descending = False) 
        elif init == "reverse":
            perm, _ = torch.sort(perm, descending = True)

        matrix = torch.zeros(self.out_channels, self.in_channels)
        if self.out_channels < self.in_channels:
            matrix[:, perm] = torch.eye(min_channels)
        else: # if self.out_channels >= self.in_channels:
            matrix[perm, :] = torch.eye(min_channels)

        weight = torch.zeros(self.out_channels, self.in_channels, *self.kernel_size)
        weight[:, :, self.kernel_size[0] // 2, self.kernel_size[1] // 2] = matrix

        self.params.data = weight

    def _project(self) -> Tensor:
        """
        Björck orthonormalization of the weights.

        """
        with torch.no_grad():
            scale, self.v_vec.data = _power_iteration_conv2d_circ(
                self.params, self.power_iters, self.power_thres, self.v_vec)

        self.weight = _bjorck_orthonormalize_conv2d_circ(
            self.params / scale, self.bjorck_order, self.bjorck_iters, self.bjorck_thres)

    def forward(self, inputs: Tensor, inverse: bool = False, cache: bool = False) -> Tensor:
        """
        Computation of the dilated 2D-convolution layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size
            [batch_size, in_channels, in_height, in_width]
            The input to the dilated 2D-convolution layer.

        inverse: bool
            Whether to computet the inverse pass of the layer.
            Default: False

        Return:
        -------
        outputs: a 4th-order tensor of size 
            [batch_size, out_channels, out_height, out_width]
            The output of the dilated 2D-convolution layer.

        """
        height, width = inputs.size(2), inputs.size(3)

        # effective dilation and kernel size
        if self.kernel_size[0] == 1 or self.kernel_size[0] >= height:
            dilation_0 = 1
            kernel_size_0 = self.kernel_size[0]
        else: # if self.kernel_size[0] > 1 and self.kernel_size[0] < height:
            dilation_0 = (height - 1) // self.kernel_size[0] + 1
            kernel_size_0 = dilation_0 * (self.kernel_size[0] - 1) + 1

        if self.kernel_size[1] == 1 or self.kernel_size[1] >= width:
            dilation_1 = 1
            kernel_size_1 = self.kernel_size[1]
        else: # if self.kernel_size[1] > 1 and self.kernel_size[1] < width:
            dilation_1 = (width -  1) // self.kernel_size[1] + 1
            kernel_size_1 = dilation_1 * (self.kernel_size[1] - 1) + 1

        dilation = (dilation_0, dilation_1)
        kernel_size_ = (kernel_size_0, kernel_size_1)

        # projection during forward pass
        if self.weight is None or (self.training and not cache):
            self._project()

        # forward/inverse computation
        if not inverse:
            outputs = F.conv2d(_circular_pad(inputs, 
                (kernel_size_[1] // 2, (kernel_size_[1] - 1) // 2,
                 kernel_size_[0] // 2, (kernel_size_[0] - 1) // 2)),
                 self.weight, self.bias, dilation = dilation)

        else: # if inverse:
            if self.bias:
                inputs = inputs - self.bias.view(-1, 1, 1)

            outputs = F.conv_transpose2d(_circular_pad(inputs,
                ((kernel_size_[1] - 1) // 2, kernel_size_[1] // 2,
                 (kernel_size_[0] - 1) // 2, kernel_size_[0] // 2)), 
                self.weight, dilation = dilation, padding = (kernel_size_[0] - 1, kernel_size_[1] - 1))

        return outputs