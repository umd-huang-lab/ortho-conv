import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single

from utils import _circular_pad, _bjorck_orthonormalize_conv1d

from torch import Tensor
from typing import Callable


class BjorckConv1d(nn.Module):

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        bias: bool = False,
        init: str = "permutation",
        iters: int = 500,
        thres: float = 1e-6
    ) -> None:
        """
        Orthogonal 1D-convolution layer with Björck orthonormalization.

        Arguments:
        ----------
        [hyper-parameters for 1D-convolution layer]
        in_channels: int
            The number of input channels to the Björck 1D-convolution layer.
        out_channels: int
            The number of output channels of the Björck 1D-convolution layer.
        kernel_size: int
            The length of the convolutional kernels.
        bias: bool
            Whether to add a learnable bias to the output.
            Default: False

        init: str ("identical", "reverse", "permutation")
            The initialization method of the orthogonal layer.
            Default: "permutation"

        [hyper-parameters for Björck orthonormalization]
        iters: int
            The maximum iterations for Björck orthonormalization.
            Default: 100
        thres: float
            The absolute tolerance for Björck orthonormalization.
            Default: 1e-6

        """
        super(BjorckConv1d, self).__init__()

        ## [hyperparameters for the Björck 1D-convolution layer]
        self.in_channels  =  in_channels
        self.out_channels = out_channels

        if self.in_channels > self.out_channels:
            warnings.warn("The layer is made row-orthogonal.")
        if self.in_channels < self.out_channels:
            warnings.warn("The layer is made column-orthogonal.")

        self.kernel_size = _single(kernel_size)

        self.iters = iters
        self.thres = thres

        ## [initialization of the layer parameters]
        self.weight = nn.Parameter(torch.zeros(
            self.out_channels, self.in_channels, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else: # if bias:
            self.register_parameter('bias', None)

        self._initialize(init)

        ## [functionals for forward/inverse computation]

        # forward computation
        _pad_forward = lambda inputs: _circular_pad(inputs,
            (self.kernel_size[0] // 2, (self.kernel_size[0] - 1) // 2))

        self._forward = lambda inputs, weights, biases: \
            F.conv1d(_pad_forward(inputs), weights, biases)

        # inverse computation
        _pad_inverse = lambda inputs: _circular_pad(inputs,
            ((self.kernel_size[0] - 1) // 2, self.kernel_size[0] // 2))

        if bias:
            self._inverse = lambda inputs, weights, biases: \
                F.conv_transpose1d(_pad_inverse(inputs - biases.view(-1, 1)),
                weights, padding = self.kernel_size[0] - 1)
        else: # if not bias:
            self._inverse = lambda inputs, weights, biases: \
                F.conv_transpose1d(_pad_inverse(inputs),
                weights, padding = self.kernel_size[0] - 1)

    def _initialize(self, init: str = "permutation") -> None:
        """
        Initialization of Björck 1D-convolution layer.

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

        weights = torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0])
        weights[:, :, self.kernel_size[0] // 2] = matrix

        self.weight.data = weights

    def _project(self) -> None:
        """
        Björck orthonormalization of the weights.

        """
        with torch.no_grad():
            weights = self.weight.data
            weights = _bjorck_orthonormalize_conv1d(weights, self.iters, self.thres)
            self.weight.data = weights

    def forward(self, inputs: Tensor, projection: bool = True, inverse: bool = False) -> Tensor:
        """
        Computation of the Björck 1D-convolution layer.

        Arguments:
        ----------
        inputs: a 3rd-order tensor of size
            [batch_size, in_channels, in_length]
            The input to the Björck 1D-convolution layer.

        projection: bool
            Whether to project the weights during forward pass.
            Default: True

        inverse: bool
            Whether to computet the inverse of the layer.
            Default: False

        Return:
        -------
        outputs: a 3rd-order tensor of size 
            [batch_size, out_channels, out_length]
            The output of the Björck 1D-convolution layer.

        """
        if self.training and projection: self._project()

        if not inverse:
            outputs = self._forward(inputs, self.weight, self.bias)
        else: # if inverse:
            outputs = self._inverse(inputs, self.weight, self.bias)

        return outputs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Testing orthogonal 1D-convolution layer
    print("Testing BjorckConv1d layer.")

    # hyper-parameters
    batch_size, kernel_size, in_length = 1, 3, 60
    iters, thres = 100, 1e-5

    # exact orthogonal, row-orthogonal, column-orthogonal
    for (in_channels, out_channels) in [(16, 16), (16, 24), (16, 12)]:
        for init in ["identical", "reverse", "permutation"]:
            print("out_channels = %d, in_channels = %d, kernel_size = %d" 
            % (out_channels,      in_channels,      kernel_size))
            print("initialization: ", init)

            # evaluate the layer with randomized inputs
            inputs  = torch.randn(batch_size, in_channels, in_length, requires_grad = True).to(device)
            module  = BjorckConv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, 
                bias = False, init = init, iters = iters, thres = thres).to(device)
            outputs = module(inputs)

            # check output dimensions
            output_size = outputs.size()
            print("output size: ", output_size)
            assert output_size == (batch_size, out_channels, in_length)

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
                grad_outputs = torch.randn(batch_size, out_channels, in_length).to(device)
                grad_inputs  = torch.autograd.grad(outputs, inputs, grad_outputs)[0]
                norm_grad_inputs  = torch.norm(grad_inputs)
                norm_grad_outputs = torch.norm(grad_outputs)

                if torch.isclose(norm_grad_outputs, norm_grad_inputs, rtol = 1e-3, atol = 1e-4):
                    print("Success! The input gradient norm matches the output gradient norm!")
                else: 
                    print("norm_grad_inputs: %.4f, norm_grad_outputs: %.4f" % 
                        (norm_grad_inputs.item(), norm_grad_outputs.item()))
                    print("The input gradient norm and output gradient norm do not match.")

            # 3) check orothogonality by reversion
            if out_channels >= in_channels: # column-orthogonal
                module.eval()
                inputs_ = module(outputs, projection = False, inverse = True)

                input_size = inputs_.size()
                assert input_size == (batch_size, in_channels, in_length)

                if torch.isclose(torch.norm(inputs - inputs_), torch.tensor(0.).to(device), rtol = 5e-3, atol = 5e-4):
                    print("Success! The restored input matches the original input.")
                else:
                    print("The restored input and the original input do not match.")
                    print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

            print('-' * 60)