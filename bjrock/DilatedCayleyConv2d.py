import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils import _circular_pad, _power_iteration_conv2d_circ, _bjorck_orthonormalize_conv2d_circ

from torch import Tensor
from typing import Callable


class DilatedBjorckConv2d(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        bias: bool = False,
        init: str = "permutation",
        steps: int = 10,
        order: int = 1,
        iters: int = 20,
        thres: float = 1e-6
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

        [hyper-parameters for Björck orthonormalization]
        steps: int
            The number of steps of power iterations before Björck orthonormalization.
            Default: 10

        order: int
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

        self.steps = steps
        self.order = order
        self.iters = iters
        self.thres = thres

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
            scale, self.v_vec.data = _power_iteration_conv2d_circ(self.params, self.steps, self.thres, self.v_vec)

        self.weight = _bjorck_orthonormalize_conv2d_circ(self.params / scale, self.order, self.iters, self.thres)

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Testing orthogonal 2D-convolution layer
    print("Testing DilatedBjorckConv1d layer...")
    print('*' * 60)

    # hyper-parameters
    batch_size, in_height, in_width = 1, 60, 60
    steps, order, iters, thres = 10, 1, 20, 1e-6
    
    for kernel_size in [(1, 1), (1, 2), (3, 2), (3, 4), (5, 5)]:
        # exact orthogonal, row-orthogonal, column-orthogonal
        for (in_channels, out_channels) in [(32, 32), (32, 24), (32, 48)]:
            print("out_channels = %d, in_channels = %d, kernel_size = %s" 
                % (out_channels,      in_channels,      kernel_size))

            # evaluate the layer with randomized inputs
            inputs  = torch.randn(batch_size, in_channels, in_height, in_width, requires_grad = True).to(device)
            module  = DilatedBjorckConv2d(in_channels = in_channels, out_channels = out_channels,
                kernel_size = kernel_size, steps = steps, order = order, iters = iters, thres = thres).to(device)
            outputs = module(inputs)

            # check output dimensions
            output_size = outputs.size()
            print("output size: ", output_size)
            assert output_size == (batch_size, out_channels, in_height, in_width)

            # 1) check forward norm preservation
            if out_channels >= in_channels: # column-orthogonal
                norm_inputs  = torch.norm(inputs) 
                norm_outputs = torch.norm(outputs)

            if torch.isclose(norm_inputs, norm_outputs, rtol = 1e-3, atol = 1e-4):
                print("Success! The input norm matches the output norm.")
            else:
                print("norm_inputs: %.4f, norm_outputs: %.4f" % 
                    (norm_inputs.item(), norm_outputs.item()))
                print("The input norm and output norm do not match.") 

            # 2) check backward norm preservation
            if out_channels <= in_channels: # row-orthogonal
                grad_outputs = torch.randn(batch_size, out_channels, in_height, in_width).to(device)
                grad_inputs  = torch.autograd.grad(outputs, inputs, grad_outputs)[0]
                norm_grad_inputs  = torch.norm(grad_inputs)
                norm_grad_outputs = torch.norm(grad_outputs)

                if torch.isclose(norm_grad_outputs, norm_grad_inputs, rtol = 1e-3, atol = 1e-4):
                    print("Success! The input gradient norm matches the output gradient norm!")
                else: 
                    print("norm_grad_inputs: %.4f, norm_grad_outputs: %.4f" % 
                        (norm_grad_inputs.item(), norm_grad_outputs.item()))
                    "The input gradient norm and output gradient norm do not match."

            # 3) check orothogonality by reversion
            if out_channels >= in_channels: # column-orthogonal
                module.eval()
                inputs_ = module(outputs, inverse = True, cache = True)

                input_size = inputs_.size()
                assert input_size == (batch_size, in_channels, in_height, in_width)

                if torch.isclose(torch.norm(inputs - inputs_), torch.tensor(0.).to(device), rtol = 5e-3, atol = 5e-4):
                    print("Success! The restored input matches the original input.")
                else:
                    print("The restored input and the original input do not match.")
                    print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

            print('-' * 60)
