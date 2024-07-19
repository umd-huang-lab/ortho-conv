import warnings

import math
import torch
import geotorch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

from .container import Matrix
from .circular_padding import _circular_pad


class OrthoConv2d(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 dilation = 1,
                 padding_mode = "circular",
                 padding = "auto",
                 groups = 1,
                 bias = False,
                 triv = "expm",
                 init = "uniform",
        ):
        """
        Construction of an orthogonal 2D-convolutional layer.

        Arguments:
        ----------
        [hyperparameters for input/output channels]
        in_channels: int
            The number of input channels to the orthogonal convolutional layer.
        out_channels: int
            The number of output channels of the orthogonal convolutional layer.
        Note: For orthogonality, out_channels == in_channels * stride[0] * stride[1] 
            If in_channels * stride[0] * stride[1] < out_channels,
                the layer is column-orthogonal (input norm perserving).
            If in_channels * stride[0] * stride[1] > out_channels, 
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
        kernel_size: int or tuple(int, int)
            The size of the convolutional kernels.
        stride: int or tuple(int, int)
            The stride of convolutional kernels.
            Default: 1
        dilation: int or tuple(int, int)
            The spacing between the elements in the convolutional kernels.
            Default: 1
        Note 1: Given an integer n, it will be repeated into a tuple (n, n).
            for example, if kernel_size = 3, it will be repeated into (3, 3).
        Note 2: For orthogonality, the kernel_size should be divsible by the stride for each dimension.
            for example, if stride[i] = 2, the kernel_size[i] can be 2, 4, 6, etc.
            kernel_size_per_stride[i] = kernel_size[i] / stride[i] for i = 0, 1
        Note 3: For orthogonality, only one of stride and dilation can be greater than one for each dimension.
            for example, dilation[i] = 2 and stride[i] = 2 is not allowed for i = 0, 1.
        
        padding_mode: str ("zeros", "reflect", "replicate" or "circular")
            The padding mode of the convolutional layer.
            Note 1: For orthogonality, the padding mode should either be "circular" or "zeros".
            Note 2: For reversibility, the padding mode should be "circular".
            Default: "circular"
        padding: str, int or tuple(int, int)
            The number padding entries added to both side of the input.
            Note: Given an integer p, it will be repeated into a tuple (p, p).
            Default: "auto", i.e., set padding automatically to satisfy orthogonality.

        [hyperparameters for the trivialization and initialization]
        triv: str ("expm" or "cayley")
            The retraction that maps a skew-symmetric matrix to an orthogonal matrix.
            Default: "expm"
        init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of the orthogonal convolutional layer.
            Default: "uniform"

        """
        super(OrthoConv2d, self).__init__()

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
        self.kernel_size = utils._pair(kernel_size)
        self.dilation = utils._pair(dilation)
        self.stride = utils._pair(stride)

        # effective kernel size
        kernel_size_ = tuple([self.dilation[i] * (self.kernel_size[i] - 1) + 1 for i in [0, 1]])

        # left/right margins (for right-tilted filters)
        kernel_margin = tuple([((kernel_size_[i] - 1) // 2, kernel_size_[i] // 2) for i in [0, 1]])

        # effective input/maximum channels for each group
        self.in_channels_per_group_ = self.in_channels_per_group * (self.stride[0] * self.stride[1])
        self.max_channels_per_group = max(self.out_channels_per_group, self.in_channels_per_group_)

        if self.in_channels_per_group_ > self.out_channels_per_group:
            warnings.warn("The input dimension is higher than the output one; The layer is made row-orthogonal.")
        elif self.in_channels_per_group_ < self.out_channels_per_group:
            warnings.warn("The input dimension is lower than the output one; The layer is made column-orthogonal.")

        # check orthogonality based on stride and dilation
        assert max([min(self.dilation[i], self.stride[i]) for i in [0, 1]]) == 1, \
            "For each dimension, only one of dilation and stride can be greater than one."

        # effective kernel size for each polyphase
        assert max([self.kernel_size[i] % self.stride[i] for i in [0, 1]]) == 0, \
            "For each dimension, the kernel size should be divisble by the stride."

        self.kernel_size_per_stride = tuple([self.kernel_size[i] // self.stride[i] for i in [0, 1]])

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

            for axis in range(2):
                if max(self.kernel_size_per_stride) == 1:
                    if axis == 1: break
                    axis_id = group_id
                else: # if max(self.kernel_size_per_stride) > 1:
                    axis_id = "%s_d%d" % (group_id, axis)

                # the orthogonal matrices are stored in Matrix containers
                self.module[axis_id] = Matrix(self.max_channels_per_group, self.max_channels_per_group)
                geotorch.orthogonal(self.module[axis_id], tensor_name = "matrix", triv = triv)

                # since F.conv2d computes correlation, the filters are right-tilted
                for pos in range(2, self.kernel_size_per_stride[axis] + 1):
                    pos_id = "%s_%s%d" % (axis_id, "l" if pos % 2 else "r", pos // 2)
                        
                    self.module[pos_id] = Matrix(self.max_channels_per_group // 2, self.max_channels_per_group)
                    geotorch.orthogonal(self.module[pos_id], tensor_name = "matrix", triv = triv)

        # parameter for the optional bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            warnings.warn("The layer is not orthogonal if bias is used.")
        else: # if not bias:
            self.register_parameter('bias', None)

        # initialize the layer parameters
        assert init in ["uniform", "torus", "identical", "reverse", "permutation"], \
            "The initialization method is not supported for orthogonal convolutional layer."

        assert init == "uniform" or sum([self.kernel_size_per_stride[i] % 2 for i in [0, 1]]) == 2, \
            "The layer cannot reduce to a matrix if both kernel_size_per_stride are odd."

        self.reset_parameters(init = init)

        ## [functionals for forward/reverse computations]

        # 1) forward computation function
        if padding == "auto":

            if padding_mode == "circular":
                shift = tuple([self.stride[i] // 2 for i in [0, 1]])

                input_padding = tuple([
                    kernel_margin[i][1] - shift[i], 
                    kernel_margin[i][0] + shift[i] - self.stride[i] + 1]
                for i in [0, 1])

            elif padding_mode == "zeros":
                shift = tuple([kernel_margin[i][1] for i in [0, 1]])

                input_padding = tuple([
                    (kernel_margin[i][1] - shift[i])  + self.stride[i] * (
                    (kernel_margin[i][0] + shift[i]) // self.stride[i]),
                    (kernel_margin[i][0] + shift[i] - self.stride[i] + 1)  + self.stride[i] * (
                    (kernel_margin[i][1] - shift[i] + self.stride[i] - 1) // self.stride[i])]
                for i in [0, 1])

            else: # if self.padding_mode in ["replicate", "reflect"]:
                raise NotImplementedError("Automatic padding is not supported for \"%s\"." % padding_mode)
        
        else: # if padding != "auto":
            padding = utils._pair(padding)
            input_padding = tuple([utils._pair(padding[i]) for i in [0, 1]])
            warnings.warn("The layer may not be orthogonal with specified padding.")

        input_padding = input_padding[1] + input_padding[0]

        self.conv_forward = lambda inputs, weights, bias: F.conv2d(_pad(inputs,
            input_padding), weights, bias, self.stride, 0, self.dilation, self.groups)
                
        # 2) reverse computation function
        if padding == "auto" and padding_mode == "circular":

            output_padding = tuple([(
                kernel_margin[i][0] + shift[i],
                kernel_margin[i][1] - shift[i] + self.stride[i] - 1)
            for i in [0, 1]])

            circular_padding = tuple([tuple([
                output_padding[i][j] // self.stride[i]
            for j in [0, 1]]) for i in [0, 1]])

            zeros_padding = tuple([tuple([
                output_padding[i][j] - self.stride[i] * circular_padding[i][j]
            for j in [0, 1]]) for i in [0, 1]])

            cropping = tuple([tuple([
                zeros_padding[i][j] - max(zeros_padding[i])
            for j in [0, 1]]) for i in [0, 1]])

            padding_ = tuple([kernel_size_[i] - 1 - max(zeros_padding[i]) for i in [0, 1]])

            cropping = cropping[1] + cropping[0]

            if min(cropping) == 0:
                _crop = lambda outputs: outputs
            else: # if min(cropping) < 0:
                _crop = lambda outputs: F.pad(outputs, cropping)

            if not bias:
                _bias = lambda outputs, vector: outputs
            else: # if bias:
                _bias = lambda outputs, vector: outputs - vector.view(-1, 1, 1)

            circular_padding = circular_padding[1] + circular_padding[0]

            self.conv_reverse = lambda outputs, weights, vector: _crop(F.conv_transpose2d(_pad(_bias(outputs,
                vector), circular_padding), weights, None, self.stride, padding_, 0, self.groups, self.dilation))

    def reset_parameters(self, init):
        """
        Initialization of the orthogonal 2D-orthogonal layer.

        Argument:
        ---------
        init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of the orthogonal 2D-convolutional layer.
            Default: "uniform"

        """

        # initialize the paraunitary matrix
        for group in range(self.groups):
            group_id = "g%d" % group

            for axis in [0, 1]:
                if max(self.kernel_size_per_stride) == 1:
                    if axis == 1: break
                    axis_id = group_id
                else: # if max(self.kernel_size_per_stride) > 1:
                    axis_id = "%s_d%d" % (group_id, axis)

                if init in ["identical", "reverse", "permutation"]:
                    matrix = torch.eye(self.max_channels_per_group)
                    if init == "reverse" and axis == 0:
                        matrix = matrix.flip(1)
                    elif init == "permutation":
                        matrix = matrix[:, torch.randperm(self.max_channels_per_group)]
                elif init == "torus":
                    matrix = self.module[axis_id].parametrizations.matrix[0].sample(init)

                if init != "uniform":
                    self.module[axis_id].matrix = matrix
                    for pos in range(2, self.kernel_size_per_stride[axis], 2):
                        self.module["%s_l%d" % (axis_id, pos // 2)].matrix = \
                            self.module[axis_id](self.module["%s_r%d" % (axis_id, pos // 2)].matrix)

        # initialize the bias to zeros
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def paraunitary(self, group):
        """
        Construction of the paraunitary matrix for the given group.

        Argument:
        ---------
        group: int
            The group identity for the paraunitary matrix.

        device: str ("cpu", "cuda", or "cuda:X")
            The device on which the weights will be allocated.

        Return:
        ------- 
        kernel: a 4th-order tensor of size 
            [out_channels_per_group, in_channels_per_group_, *kernel_size_per_stride]
            The paraunitary matrix for the given group.

        Notes:
        ------
        out_channels_per_group = out_channels / groups
        in_channels_per_group  = in_channels / groups
        in_channels_per_group_ = in_channels_per_group * stride[0] * stride[1]
        kernel_size_per_stride[i] = kernel_size[i] / stride[i] for i = 0, 1

        """
        group_id = "g%d" % group

        if max(self.kernel_size_per_stride) == 1:

            # [max_channels_per_group, max_channels_per_group, *kernel_size_per_stride]
            kernel = self.module[group_id].matrix.unsqueeze(-1).unsqueeze(-1)

        else: # if max(self.kernel_size_per_stride) > 1:
            factors = [None] * 2
            for axis in range(2):
                axis_id = "%s_d%d" % (group_id, axis)

                # [1, max_channels_per_group, max_channels_per_group]
                factor = self.module[axis_id].matrix.unsqueeze(0)
                zeros  = torch.zeros_like(factor)
                
                # [kernel_size_per_stride, max_channels_per_group, max_channels_per_group]
                for pos in range(2, self.kernel_size_per_stride[axis] + 1):
                    factor_padded_left  = torch.cat((zeros, factor), dim = 0)
                    factor_padded_right = torch.cat((factor, zeros), dim = 0)

                    if pos % 2 == 0:
                        matrix = self.module["%s_r%d" % (axis_id, pos // 2)].matrix
                        factor = factor_padded_left  + torch.einsum("kts,sr,nr->ktn",
                            factor_padded_right - factor_padded_left, matrix, matrix)
                    else: # if pos % 2 == 1:
                        matrix = self.module["%s_l%d" % (axis_id, pos // 2)].matrix
                        factor = factor_padded_right + torch.einsum("kts,tr,nr->kns",
                            factor_padded_left - factor_padded_right, matrix, matrix)

                factors[axis] = factor

            # [max_channels_per_group, max_channels_per_group, *kernel_size_per_stride]
            kernel = torch.einsum("hrs,wtr->tshw", factors)

        # [out_channels_per_group, in_channels_per_group_, *kernel_size_per_stride]
        kernel = kernel[:self.out_channels_per_group, :self.in_channels_per_group_]

        return kernel

    def weights(self):
        """
        Construction of the weights for forward or reverse computation.

        Return:
        -------
        weights: a 4th-order tensor of size
            [out_channels, in_channels_per_group, kernel_size[0], kernel_size[1]]
            The convolutional kernel for forward or reverse computation.

        Notes:
        ------
        The weights tensor is a concatenation of [groups] 4th-order tensors of size
            [out_channels_per_group, in_channels_per_group, kernel_size[0], kernel_size[1]] over output channels.
            out_channels_per_group = out_channels / groups
             in_channels_per_group =  in_channels / groups

        """
        kernels = [None] * self.groups

        for group in range(self.groups):

            # [out_channels_per_group, in_channels_per_group_, kernel_size_per_stride[0], kernel_size_per_stride[1]]
            kernel = self.paraunitary(group)

            if sum(self.stride) > 2:
                # [out_channels_per_group, in_channels_per_group, stride[0], stride[1], kernel_size_per_stride[0], kernel_size_per_stride[1]]
                kernel = kernel.view(self.out_channels_per_group, self.in_channels_per_group, 
                    self.stride[0], self.stride[1], self.kernel_size_per_stride[0], self.kernel_size_per_stride[1])

                # [out_channels_per_group, in_channels_per_group, kernel_size_per_stride[0], stride[0], kernel_size_per_stride[1], stride[1]]
                kernel = kernel.permute(0, 1, 4, 2, 5, 3).contiguous()

                # [out_channels_per_group, in_channels_per_group, kernel_size[0], kernel_size[1]]
                kernel = kernel.view(self.out_channels_per_group, self.in_channels_per_group, 
                    self.kernel_size_per_stride[0] * self.stride[0], self.kernel_size_per_stride[1] * self.stride[1])

            kernels[group] = kernel

        # concatenate the kernels from different groups over the output channels
        # weights: [out_channels, in_channels_per_groups, kernel_size[0], kernel_size[1]]
        weights = torch.cat(kernels, dim = 0)

        return weights

    def forward(self, inputs, reverse = False, cache = False):
        """
        Computation of the orthogonal 2D-convolutional layer. 

        Argument:
        ---------
        inputs: a 4th-order tensor of size 
            [batch_size,  in_channels,  in_height,  in_width]
            The input to the orthogonal 2D-convolutional layer.

        reverse: bool
            Whether to compute the layer's reverse pass.
            Default: False    

        cache: bool
            Whether to use the cached weights for computation.
            Default: False

        Return:
        -------
        outputs: a 4th-order tensor of size 
            [batch_size, out_channels, out_length, out_width]
            The output of the orthgonal 2D-convolutional layer.

        Notes:
        ------
        For orthogonality, out_channels = in_channels * stride[0] * stride[1]
            If padding_mode == "circular":
                out_height = in_height / stride[0]
                out_width  = in_height / stride[1]
            If padding_mode == "zeros":
                out_height = (in_height + dilation[0] * (kernel_size[0] - 1) / stride[0]
                out_width  = (in_width  + dilation[1] * (kernel_size[1] - 1) / stride[1]

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

    # orthogonal 2D-convolutional layer
    print("Testing orthogonal 2D-convolutional layer")
    
    # batch size and input dimensions
    batch_size, in_height, in_width = 2, 60, 60

    # dilated or strided convolutions
    for (dilation, stride) in [((1, 1), (1, 1)), ((2, 3), (1, 1)), ((3, 2), (1, 1)), ((1, 1), (3, 2)), ((1, 1), (3, 2))]:

        # exact orthogonal, row-orthogonal, and column-orthogonal
        for (in_channels, out_channels_per_stride) in [(32, 32), (32, 24), (32, 48)]:
            out_channels = out_channels_per_stride * stride[0] * stride[1]

            # effective kernel size for each polyphase
            for kernel_size_per_stride in [(1, 1), (1, 3), (3, 1), (3, 3)]:
                kernel_size = (kernel_size_per_stride[0] * stride[0], kernel_size_per_stride[1] * stride[1])

                # padding mode for orthogonality
                for padding_mode in ["circular", "zeros"]:
                    print("out_channels = %d, kernel_size = %s, stride = %s, dilation = %s, padding_mode = %s" 
                        % (out_channels, kernel_size, stride, dilation, padding_mode))

                    # compute output height/width according to padding mode
                    if padding_mode == "circular":
                        out_height = in_height // stride[0]
                        out_width  = in_height // stride[1]
                    elif padding_mode == "zeros":
                        out_height = (in_height + dilation[0] * (kernel_size[0] - 1)) // stride[0]
                        out_width  = (in_width  + dilation[1] * (kernel_size[1] - 1)) // stride[1]

                    # initialize the orthogonal 2D-convolutional layer
                    layer = OrthoConv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                            dilation = dilation, stride = stride, padding_mode = padding_mode, padding = "auto").to(device)

                    # evaluate the layer using randomized input
                    inputs  = torch.randn(batch_size, in_channels, in_height, in_width, requires_grad = True).to(device)
                    outputs = layer(inputs, reverse = False, cache = False)

                    # check forward norm preservation
                    if out_channels >= in_channels * stride[0] * stride[1]: # column-orthogonal
                        norm_inputs  = torch.norm(inputs)
                        norm_outputs = torch.norm(outputs)

                        if not torch.isclose(norm_inputs, norm_outputs, rtol = 1e-3, atol = 1e-4):
                            print("norm_inputs: %.4f, norm_outputs: %.4f" % 
                                (norm_inputs.item(), norm_outputs.item()))
                            print("The input norm and output norm do not match.")

                    if padding_mode == "circular":

                        # check backward norm preservation
                        if out_channels <= in_channels * stride[0] * stride[1]: # row-orthogonal
                            grad_outputs = torch.randn(batch_size, out_channels, out_height, out_width).to(device)
                            grad_inputs  = torch.autograd.grad(outputs, inputs, grad_outputs)[0]

                            norm_grad_inputs  = torch.norm(grad_inputs)
                            norm_grad_outputs = torch.norm(grad_outputs)

                            if not torch.isclose(norm_grad_outputs, norm_grad_inputs, rtol = 1e-3, atol = 1e-4):
                                print("norm_grad_inputs: %.4f, norm_grad_outputs: %.4f" % 
                                    (norm_grad_inputs.item(), norm_grad_outputs.item()))
                                print("The input gradient norm and output gradient norm do not match.")

                        # check orothogonality by reversion
                        if out_channels >= in_channels * stride[0] * stride[1]: # column-orthogonal
                            inputs_ = layer(outputs, reverse = True, cache = True)

                            if torch.isclose(torch.norm(inputs - inputs_), torch.tensor(0.).to(device), rtol = 1e-2, atol = 1e-3):
                                print("Success! The restored input matches the original input.")
                            else:
                                print("The restored input and the original input do not match.")
                                print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

                    print('-' * 60)
