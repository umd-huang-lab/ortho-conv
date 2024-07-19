import warnings

import torch
import geotorch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

from torch import Tensor
from typing import Optional, List, Tuple, Union

from utilities import Matrix, _circular_pad


class OrthoConvTranspose1d(nn.Module):
    
    def __init__(
        self,
        in_channels:  int, 
        out_channels: int, 
        kernel_size: Union[int, Tuple[int]], 
        stride:      Union[int, Tuple[int]] = 1, 
        dilation:    Union[int, Tuple[int]] = 1,
        padding_mode: str = "circular",
        padding: Union[str, int, Tuple[int]] = "auto",
        output_padding: Union[int, Tuple[int]] = 0,
        groups: int = 1,
        bias: bool = False,
        triv: str = "expm",
        init: str = "permutation"
    ) -> None:
        """
        Construction of an orthogonal transposed 1D-convolutional layer.

        Arguments:
        ----------
        [hyperparameters for input/output channels]
        in_channels: int
            The number of input channels to the 1D-orthogonal layer.
        out_channels: int
            The number of output channels of the 1D-orthogonal layer.
        Note: For orthogonality, in_channels == out_channels * stride
            If in_channels < out_channels * stride,
                the layer is column-orthogonal (input norm perserving).
            If in_channels > out_channels * stride, 
                the layer is row-orthogonal (gradient norm perserving).

        groups: int
            The number of interconnected blocks in the convolutional layer.
            Note: Both in_channels and out_channels should be divisible by groups.
                 in_channels_per_group =  in_channels / groups
                out_channels_per_group = out_channels / groups
            Default: 1
        bias: bool
            Whether to add a learnable bias to the output.
            Note: For orthogonality, the learnable bias should be disabled.
            Default: False

        [hyperparameters for each filter and boundary condition]
        kernel_size: int or a tuple of (int, )
            The length of the convolutional kernels.
        stride: int or a tuple of (int, )
            The stride of convolutional kernels.
            Default: 1
        dilation: int or a tuple of (int, )
            The spacing between the elements in the convolutional kernels.
            Default: 1
        Note 1: For orthogonality, the kernel_size should be divisble by the stride.
            kernel_size_per_stride = kernel_size / stride
            for example, if stride = 2, the kernel_size can be 2, 4, 6, etc.
        Note 2: For orthogonality, only one of stride and dilation can be greater than one.
            for example, dilation = 2 and stride = 2 is not allowed.
        
        padding_mode: str ("zeros", "reflect", "replicate", or "circular")
            The padding mode of the convolutional layer.
            Note 1: For orthogonality, the padding mode should either be "circular" or "zeros".
            Note 2: For reversibility, the padding mode should be "circular".
            Default: "circular"
        padding: str or int or tuple(int, )
            The number of padding entries added to both sides of the input.
            Default: "auto", i.e., set padding automatically to satisfy orthogonality.
        output_padding: int or tuple(int, )
            The number of additional entries added to one side of the output.
            Note: the argument is ignored if padding is "auto".
            Default: 0

        [hyperparameters for the trivialization and initialization]
        triv: str ("expm" or "cayley")
            The retraction that maps a skew-symmetric matrix to an orthogonal matrix.
            Default: "expm"
        init: str ("uniform", "torus", identical", "reverse", or "permutation")
            The initialization of the orthogonal convolutional layer.
            Default: "uniform"

        """
        super(OrthoConvTranspose1d, self).__init__()

        ## [hyperparameters for input/output channels]
        self.groups = groups

        assert in_channels  % self.groups == 0, \
            "The input channels should be divisble by the number of groups."
        assert out_channels % self.groups == 0, \
            "The output channels should be divisble by the number of groups."

        # number of input and output channels for each group 
        self.in_channels_per_group  =  in_channels // groups
        self.out_channels_per_group = out_channels // groups

        ## [hyperparameters for each filter and boundary condition]
        self.kernel_size = utils._single(kernel_size)
        self.dilation = utils._single(dilation)
        self.stride = utils._single(stride)

        # effective kernel size
        kernel_size_ = self.dilation[0] * (self.kernel_size[0] - 1) + 1

        # left/right margins (for left-tilted filters)
        kernel_margins = (kernel_size_ // 2, (kernel_size_ - 1) // 2)

        # effective output channels for each group
        self.out_channels_per_group_ = self.out_channels_per_group * self.stride[0]
        self.max_channels_per_group  = max(self.in_channels_per_group, self.out_channels_per_group_)

        # check orthogonality based on effective input channnels
        if self.in_channels_per_group > self.out_channels_per_group_:
            warnings.warn("The layer is made row-orthogonal.")
        elif self.in_channels_per_group < self.out_channels_per_group_: 
            warnings.warn("The layer is made column-orthogonal.")

        # check orthogonality based on stride and dilation
        assert min(self.dilation[0], self.stride[0]) == 1, \
            "For orthogonality, only one of dilation and stride can be greater than one."

        # effective kernel size for each polyphase
        assert self.kernel_size[0] % self.stride[0] == 0, \
            "For orthogonality, the kernel size should be divisble by the stride."
        self.kernel_size_per_stride = utils._single(self.kernel_size[0] // self.stride[0])

        # boundary condition for each convolution
        assert padding_mode in ["zeros", "reflect", "replicate", "circular"], \
            "The padding mode \"%s\" is not supported." % padding_mode

        # expanded padding function
        if padding_mode == "circular":
            _pad = lambda inputs, expanded_padding: _circular_pad(inputs, expanded_padding)
        elif padding_mode == "zeros":
            _pad = lambda inputs, expanded_padding: F.pad(inputs, expanded_padding, mode = "constant")
        else: # if padding_mode in ["reflect", "replicate"]:
            _pad = lambda inputs, expanded_padding: F.pad(inputs, expanded_padding, mode = padding_mode)

        ## [construction of the layer parameters]

        # parameters for the paraunitary matrix
        self.kernel = None
        self.module = nn.ModuleDict()

        for group in range(self.groups):
            group_id = "g%d" % group

            # the orthogonal matrices are stored in Matrix containers
            self.module[group_id] = Matrix(self.max_channels_per_group, self.max_channels_per_group)
            geotorch.orthogonal(self.module[group_id], tensor_name = "matrix", triv = triv)

            # since F.conv_transpose1d computes convolution, the filters are left-tilted
            for pos in range(2, self.kernel_size_per_stride[0] + 1):
                pos_id = "%s_%s%d" % (group_id, "r" if pos % 2 else "l", pos // 2)

                self.module[pos_id] = Matrix(self.max_channels_per_group // 2, self.max_channels_per_group)
                geotorch.orthogonal(self.module[pos_id], tensor_name = "matrix", triv = triv)

        # parameter for the optional bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else: # if not bias:
            self.register_parameter('bias', None)

        # initialize the layer parameters
        assert init in ["uniform", "torus", "identical", "reverse", "permutation"], \
            "The initialization method is not supported for orthogonal convolutioal layer."

        assert init == "uniform" or self.kernel_size_per_stride[0] % 2 == 1, \
            "The layer cannot reduce to a matrix if kernel_size_per_stride is odd."

        self.reset_parameters(init = init)

        ## [functionals for forward/reverse computations]

        # 1) forward computation
        if padding == "auto":

            if padding_mode == "circular":
                shift = self.stride[0] // 2

                input_padding = (kernel_margins[1] + shift, 
                                 kernel_margins[0] - shift + self.stride[0] - 1)

                circular_padding = tuple([input_padding[j] // self.stride[0] for j in [0, 1]])

                zeros_padding = tuple([input_padding[j] - self.stride[0] * circular_padding[j] for j in [0, 1]])

                cropping = tuple([zeros_padding[j]  - max(zeros_padding) for j in [0, 1]])

                padding_ = (kernel_size_ - 1) - max(zeros_padding)

                if min(cropping) == 0:
                    _crop = lambda outputs: outputs
                else: # if min(cropping) < 0:
                    _crop = lambda outputs: F.pad(outputs, cropping)

                self.conv_forward = lambda inputs, weights, bias: _crop(F.conv_transpose1d(_pad(inputs,
                    circular_padding), weights, bias, self.stride, padding_, 0, self.groups, self.dilation))

            elif padding_mode == "zeros":

                self.conv_forward = lambda inputs, weights, bias: F.conv_transpose1d(
                    inputs, weights, bias, self.stride, 0, 0, self.groups, self.dilation)

            else: # if padding_mode in ["reflect", "replicate"]: 
                raise NotImplementedError("Automatic padding is not supported for \"%s\"." % padding_mode)

        else: # if padding != "auto":
            warnings.warn("The layer may not be orthogonal with specified padding.")
            self.conv_forward = lambda inputs, weights, bias: F.conv_transpose1d(inputs,
                weights, bias, self.stride, padding, output_padding, self.groups, self.dilation)

        # 2) reverse computation
        if padding == "auto" and padding_mode == "circular":

            output_padding = (kernel_margins[0] - shift,
                              kernel_margins[1] + shift - self.stride[0] + 1)

            if not bias:
                _bias = lambda outputs, vector: outputs
            else: # if bias:
                _bias = lambda outputs, vector: outputs - vector.view(-1, 1)

            self.conv_reverse = lambda outputs, weights, vector: F.conv1d(_pad(_bias(outputs,
                vector), output_padding), weights, None, self.stride, 0, self.dilation, self.groups)

    def reset_parameters(self, init: str) -> None:
        """
        Initialization of the orthogonal transposed 1D-orthogonal layer.

        Argument:
        ---------
        init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of the orthogonal transposed 1D-convolutional layer.
            Default: "uniform"

        """

        # initialize the paraunitary matrix
        for group in range(self.groups):
            group_id = "g%d" % group

            if init in ["identical", "reverse", "permutation"]:
                matrix = torch.eye(self.max_channels_per_group)
                if init == "reverse":
                    matrix = matrix.flip(1)
                elif init == "permutation":
                    matrix = matrix[:, torch.randperm(self.max_channels_per_group)]
            else: # if init in ["uniform", "torus"]
                matrix = self.module[group_id].parametrizations.matrix[0].sample(init)

            self.module[group_id].matrix = matrix
            if init != "uniform":
                for pos in range(2, self.kernel_size_per_stride[0], 2):
                    self.module["%s_l%d" % (group_id, pos // 2)].matrix = \
                        self.module[group_id](self.module["%s_r%d" % (group_id, pos // 2)].matrix)

        # initialize the bias to zeros
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def paraunitary(self, group: int) -> Tensor:
        """
        Construction of the paraunitary matrix for the given group.

        Argument:
        ---------
        group: int
            The group identity for the paraunitary matrix.

        Return:
        ------- 
        kernel: a 3rd-order tensor of size 
            [in_channels_per_group, out_channels_per_group_, kernel_size_per_stride]
            The paraunitary matrix for the given group.

        Notes:
        ------
        in_channels_per_group   = in_channels / groups
        out_channels_per_group  = out_channels / groups
        out_channels_per_group_ = out_channels_per_group * stride
        kernel_size_per_stride  = kernel_size / stride

        """
        group_id = "g%d" % group

        # [1, max_channels_per_group, max_channels_per_group]
        kernel = self.module[group_id].matrix.unsqueeze(0)
        zeros  = torch.zeros_like(kernel)

        # [kernel_size_per_stride, max_channels_per_group, max_channels_per_group]
        for pos in range(2, self.kernel_size_per_stride[0] + 1):
            kernel_padded_left  = torch.cat((zeros, kernel), dim = 0)
            kernel_padded_right = torch.cat((kernel, zeros), dim = 0)

            if pos % 2 == 0:
                matrix = self.module["%s_l%d" % (group_id, pos // 2)].matrix
                kernel = kernel_padded_left  + torch.einsum("kst,sr,nr->knt",
                    kernel_padded_right - kernel_padded_left, matrix, matrix)
            else: # if pos % 2 == 1:
                matrix = self.module["%s_r%d" % (group_id, pos // 2)].matrix
                kernel = kernel_padded_right + torch.einsum("kst,tr,nr->ksn",
                    kernel_padded_left - kernel_padded_right, matrix, matrix)

        # [max_channels_per_group, max_channels_per_group, kernel_size_per_stride]
        kernel = kernel.permute(1, 2, 0)

        # [in_channels_per_group, out_channels_per_group_, kernel_size_per_stride]
        kernel = kernel[:self.in_channels_per_group, :self.out_channels_per_group_]

        return kernel

    def weights(self) -> Tensor:
        """
        Construction of the weights for forward or reverse computation.

        Return:
        -------
        weights: a 3rd-order tensor of size
            [in_channels, out_channels_per_group, kernel_size]
            The convolutional kernel for forward or reverse computation.

        Notes:
        ------
        The weights tensor is a concatenation of [groups] 3rd-order tensors of size
            [in_channels_per_group, out_channels_per_group, kernel_size] over input channels.
            out_channels_per_group = out_channels / groups
             in_channels_per_group =  in_channels / groups
        
        """
        kernels = [None] * self.groups

        for group in range(self.groups):

            # [in_channels_per_group, out_channels_per_group_, kernel_size_per_stride]
            kernel = self.paraunitary(group)
            
            if self.stride[0] > 1:
                # [in_channels_per_group, out_channels_per_group, stride, kernel_size_per_stride]
                kernel = kernel.view(self.in_channels_per_group, self.stride[0], self.out_channels_per_group, -1)

                # [in_channels_per_group, out_channels_per_group, kernel_size_per_stride, stride]
                kernel = kernel.permute(0, 2, 3, 1).contiguous()

                # [in_channels_per_group, out_channels_per_group, kernel_size]
                kernel = kernel.view(self.in_channels_per_group, self.out_channels_per_group, -1)

            kernels[group] = kernel

        # concatenate the kernels from different groups over the input channels
        # weights: [in_channels, out_channels_per_group, kernel_size]
        weights = torch.cat(kernels, dim = 0)

        return weights

    def forward(self, inputs: Tensor, reverse: bool = False, cache: bool = False) -> Tensor:
        """
        Computation of the orthogonal transposed 1D-convolutional layer. 

        Argument:
        ---------
        inputs: a 3rd-order tensor of size 
            [batch_size, in_channels, in_length]
            The input to the orthogonal transposed 1D-convolutional layer.

        reverse: bool
            Whether to compute the layer's reverse pass.
            Default: False    

        cache: bool
            Whether to use the cached weights for computation.
            Default: False

        Return:
        -------
        outputs: a 3rd-order tensor of size 
            [batch_size, out_channels, out_length]
            The output of the orthgonal transposed 1D-convolutional layer.

        Notes:
        ------
        For orthogonality, out_channels = in_channels * stride
            If padding_mode == "circular":
                out_length = in_length * stride
            If padding_mode == "zeros":
                out_length = (in_length - 1) * stride + dilation * (kernel_size - 1)

        """
        if not cache or self.kernel is None:
            self.kernel = self.weights()

        if not reverse:
            outputs = self.conv_forward(inputs, self.kernel, self.bias)
        else: # if reverse:
            outputs = self.conv_reverse(inputs, self.kernel, self.bias)

        return outputs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## orthogonal transposed 1D-convolutional layer
    print("Testing orthogonal transposed 1D-convolutional layer...")

    # batch size and input length
    batch_size, in_length = 2, 30
    
    # dilated or strided convolutions
    for (dilation, stride) in [(1, 1), 
        (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (3, 1), (4, 1), (5, 1)]:

        # exact orthogonal, row-orthogonal, column-orthogonal
        for (in_channels, out_channels_times_stride) in [(120, 120), (120, 60), (120, 180)]:
            out_channels = out_channels_times_stride // stride

            # effective kernel size for each polyphase 
            for kernel_size_per_stride in [1, 3]:
                kernel_size = kernel_size_per_stride * stride

                # padding mode for orthogonality
                for padding_mode in ["circular", "zeros"]:
                    print("in_channels = %d, out_channels = %d, kernel_size = %d, stride = %d, dilation = %d, padding_mode = %s" %
                        (in_channels, out_channels, kernel_size, stride, dilation, padding_mode))

                    # compute output length according to padding mode
                    if padding_mode == "circular":
                        out_length = in_length * stride
                    elif padding_mode == "zeros":
                        out_length = (in_length - 1) * stride + 1 + dilation * (kernel_size - 1)

                    # initialize the orthogonal transposed 1D-convolutional layer
                    layer = OrthoConvTranspose1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                            stride = stride, padding = "auto", dilation = dilation, padding_mode = padding_mode).to(device)

                    # evaluate the layer with randomized inputs
                    inputs = torch.randn(batch_size, in_channels, in_length, requires_grad = True).to(device)
                    outputs = layer(inputs, reverse = False, cache = False)

                    # check forward norm preservation
                    if out_channels * stride >= in_channels: # column-orthogonal
                        norm_inputs  = torch.norm(inputs) 
                        norm_outputs = torch.norm(outputs)

                        if not torch.isclose(norm_inputs, norm_outputs, rtol = 1e-2, atol = 1e-4):
                            print("norm_inputs: %.5f; norm_outputs: %.5f" % 
                                (norm_inputs.item(), norm_outputs.item()))
                            print("The input norm and output norm do not match.")

                    if padding_mode == "circular":

                        # check backward norm preservation
                        if out_channels * stride <= in_channels: # column-orthogonal
                            grad_outputs = torch.randn(batch_size, out_channels, out_length).to(device)
                            grad_inputs  = torch.autograd.grad(outputs, inputs, grad_outputs)[0]

                            norm_grad_inputs  = torch.norm(grad_inputs)
                            norm_grad_outputs = torch.norm(grad_outputs)

                            if not torch.isclose(norm_grad_outputs, norm_grad_inputs, rtol = 1e-2, atol = 1e-4):
                                print("norm_grad_inputs: %.5f, norm_grad_outputs: %.5f" % 
                                    (norm_grad_inputs.item(), norm_grad_outputs.item()))
                                print("The input gradient norm and output gradient norm do not match.")

                        # check orthogonal by reversion
                        if out_channels * stride >= in_channels: # row-orthogonal
                            inputs_ = layer(outputs, reverse = True, cache = True)

                            if torch.isclose(torch.norm(inputs - inputs_), torch.tensor(0.).to(device), rtol = 1e-2, atol = 1e-3):
                                print("Success! The restored input matches the original input.")
                            else:
                                print("The restored input and the original input do not match.")
                                print("norm(diff): %.5f" % (torch.norm(inputs - inputs_).item()))

                    print('-' * 60)
                    
