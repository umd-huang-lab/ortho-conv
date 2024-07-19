import warnings

import math
import torch
import geotorch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

from container import Matrix
from circular_padding import _circular_pad


class BlockConv2d(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            pattern = {},
            gain = 1,
            stride = 1,
            dilation = 1,
            padding_mode = "circular",
            padding = "auto",
            bias = False,
            triv = "expm",
            init = "torus",
        ):
        """
        Construction of a block 2D-convolutional layer.

        Arguments:
        ----------
        [hyperparameters for input/output channels]
        in_channels: a list of ints
            The input channels of the block 2D-orthogonal layer.
        out_channels: a list of ints
            The output channels of the block 2D-orthogonal layer.

        pattern: a dictionary of {(input_block, output_block): property}
            The connectivity pattern of the block 2D-orthogonal layer.
            Options: "normal", "orthogonal", "identical", "zero"
            Default: {}, i.e., all blocls are normal convolutions.

        gain: float
            The scaling factor for the initialization.
            Default: 1
        bias: bool
            Whether to add a learnable bias to the output.
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
            for example, if kernel_size = 3, it is repeated into (3, 3).
        Note 2: For orthogonality, the kernel_size should be divsible by the stride for each dimension.
            for example, if stride[i] = 2, the kernel_size[i] can be 2, 4, 6, etc.
            kernel_size_per_stride[i] = kernel_size[i] / stride[i] for i = 0, 1
        Note 3: For orthogonality, only one of stride and dilation can be greater than one for each dimension.
            for example, dilation[i] = 2 and stride[i] = 2 is not allowed for i = 0, 1.

        padding_mode: str ("zeros", "reflect", "replicate" or "circular")
            The padding mode of the convolutional layer.
            Note: For orthogonality, the padding mode is either "circular" or "zeros".
            Default: "circular"
        padding: str, int or tuple(int, int)
            The number padding entries added to both side of the input.
            Note: Given an integer p, it is repeated into a tuple (p, p).
            Default: "auto", i.e., set padding automatically to satisfy orthogonality.

        [hyperparameters for the trivialization and initialization]
        triv: str ("expm" or "cayley")
            The retraction that maps a skew-symmetric matrix to an orthogonal matrix.
            Default: "expm"
        init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of the orthogonal convolutional layer.
            Default: "uniform"

        """
        super(BlockConv2d, self).__init__()

        ## [hyperparameters for input/output channels
        self.in_channels  = in_channels
        self.out_channels = out_channels

        ## [hyperparameters for each filter and boundary condition]
        self.kernel_size = utils._pair(kernel_size)
        self.dilation = utils._pair(dilation)
        self.stride = utils._pair(stride)

        # effective input channels for each block
        self.in_channels_ = tuple([
            self.in_channels[b] * self.stride[0] * self.stride[1]
        for b in range(len(self.in_channels))])

        # effective kernel size for each dimension
        kernel_size_ = tuple([
            self.dilation[i] * (self.kernel_size[i] - 1) + 1 
        for i in range(2)])

        # left/right margins for each dimension
        kernel_margin = tuple([
            ((kernel_size_[i] - 1) // 2, kernel_size_[i] // 2) 
        for i in range(2)])

        # check orthogonality based on stride and dilation
        assert max([min(self.dilation[i], self.stride[i]) for i in range(2)]) == 1, \
            "For each dimension, only one of dilation and stride can be greater than one."

        # effective kernel size for each polyphase
        assert max([self.kernel_size[i] % self.stride[i] for i in range(2)]) == 0, \
            "For each dimension, the kernel size should be divisble by the stride."

        self.kernel_size_per_stride = tuple([self.kernel_size[i] // self.stride[i] for i in range(2)])

        # boundary condition for each convolution
        assert padding_mode in ["zeros", "reflect",  "replicate", "circular"], \
            "The padding mode \"%s\" is not supported." % padding_mode

        # connectivity pattern for each block
        self.pattern = pattern
        self.max_channels = {}

        for in_block in range(len(self.in_channels)):
            for out_block in range(len(self.out_channels)):
                block_id =   "g%d-%d" % (in_block, out_block)

                if (in_block, out_block) in self.pattern:
                    assert self.pattern[(in_block, out_block)] \
                    in ["normal", "orthogonal", "identical", "zero"], \
                        "The connectivity pattern is not supported." 
                else: # if (in_block, out_block) not in pattern:
                    self.pattern[(in_block, out_block)] = "normal"

                self.max_channels[block_id] = max(
                    self.out_channels[out_block], self.in_channels_[in_block])

        ## [construction of the layer parameters]

        # parameters for each convolutional block
        self.kernel = None
        self.module = nn.ModuleDict()
        self.params = nn.ParameterDict()

        for in_block in range(len(in_channels)):
            for out_block in range(len(out_channels)):
                block_id =   "g%d-%d" % (in_block, out_block)

                tensor = torch.zeros(out_channels[out_block],
                    in_channels[in_block], self.kernel_size[0], self.kernel_size[1])

                if self.pattern[(in_block, out_block)] == "normal":
                    self.params[block_id] = nn.Parameter(tensor, requires_grad = True)

                elif self.pattern[(in_block, out_block)] == "zero":
                    self.params[block_id] = nn.Parameter(tensor, requires_grad = False)

                elif self.pattern[(in_block, out_block)] == "identical":
                    assert max(self.stride) == 1 and max(self.dilation) == 1, \
                        "A strided or dilated convolutional layer can not have identical block."
                    assert self.in_channels[in_block] == self.out_channels[out_block], \
                        "The input and output channels of an identical block can not be different."

                    tensor[:, :, kernel_margin[0][1], kernel_margin[1][1]] = torch.eye(self.in_channels[in_block])
                    self.params[block_id] = nn.Parameter(tensor, requires_grad = False)

                else: # if self.pattern[(in_block, out_block)] == "orthogonal":
                    max_channels = self.max_channels[block_id]

                    for axis in range(2):
                        if max(self.kernel_size_per_stride) == 1:
                            if axis == 1: break
                            axis_id = block_id
                        else: # if max(self.kernel_size_per_stride) > 1:
                            axis_id = "%s_d%d" % (block_id, axis)

                        # the orthogonal matrices are stored in Matrix containers
                        self.module[axis_id] = Matrix(max_channels, max_channels)
                        geotorch.orthogonal(self.module[axis_id], "matrix", triv = triv)

                        # since F.conv2d computes correlation, the filters are right-tilted
                        for pos in range(2, self.kernel_size_per_stride[axis] + 1):
                            pos_id = "%s_%s%d" % (axis_id, "l" if pos % 2 else "r", pos // 2)

                            self.module[pos_id] = Matrix(max_channels // 2, max_channels)
                            geotorch.orthogonal(self.module[pos_id], "matrix", triv = triv)

        # parameter for the optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(sum(out_channels)))
        else: # if not bias:
            self.register_parameter('bias', None)

        # initialize the layer parameters
        assert init in ["uniform", "torus", "identical", "reverse", "permutation"], \
            "The initialization method is not supported for orthogonal convolutional layer."

        assert init == "uniform" or max([self.kernel_size_per_stride[axis] % 2 for axis in range(2)]) == 1, \
            "The layer cannot reduce to a matrix if both kernel_size_per_stride are odd."

        self.reset_parameters(init = init, gain = gain)

        ## [functionals for forward computations]

        # extended padding functon
        if padding == "auto":

            if padding_mode == "circular":
                shift = tuple([self.stride[i] // 2 for i in range(2)])

                padding = tuple([
                    kernel_margin[i][1] - shift[i], 
                    kernel_margin[i][0] + shift[i] - self.stride[i] + 1]
                for i in range(2))

            elif padding_mode == "zeros":
                shift = tuple([kernel_margin[i][1] for i in range(2)])

                padding = tuple([
                    (kernel_margin[i][1] - shift[i])  + self.stride[i] * (
                    (kernel_margin[i][0] + shift[i]) // self.stride[i]),
                    (kernel_margin[i][0] + shift[i] - self.stride[i] + 1)  + self.stride[i] * (
                    (kernel_margin[i][1] - shift[i] + self.stride[i] - 1) // self.stride[i])]
                for i in range(2))

            else: # if self.padding_mode in ["replicate", "reflect"]:
                raise NotImplementedError("Automatic padding is not supported for \"%s\"." % padding_mode)
        
        else: # if padding != "auto":
            padding = utils._pair(padding)
            padding = tuple([utils._pair(padding[i]) for i in range(2)])
            warnings.warn("The layer may not be orthogonal with specified padding.")

        padding = padding[1] + padding[0]

        if padding_mode == "circular":
            _pad = lambda inputs: _circular_pad(inputs, padding)
        elif padding_mode == "zeros":
            _pad = lambda inputs: F.pad(inputs, padding, mode = "constant")
        else: # if padding_mode in ["reflect", "replicate"]:
            _pad = lambda inputs: F.pad(inputs, padding, mode = padding_mode)

        # forward computation function
        self.conv = lambda inputs, weights, bias: F.conv2d(
            _pad(inputs), weights, bias, self.stride, 0, self.dilation)

    def reset_parameters(self, init, gain):
        """
        Initialization of the block 2D-orthogonal layer.

        Argument:
        ---------
        init: str 
            The initialization of the orthogonal convolutional layer.
            Options: "uniform", "torus", "identical", "reverse", or "permutation"
            Default: "uniform"

        gain: float
            The scaling factor for the initialization.
            Default: 1

        """

        # initialize each block of the layer
        for in_block in range(len(self.in_channels)):
            for out_block in range(len(self.out_channels)):
                block_id = "g%d-%d" % (in_block, out_block)

                if self.pattern[(in_block, out_block)] == "normal":
                    fan_in  = self.kernel_size[0] * self.kernel_size[1] * sum(self.in_channels)
                    bound = gain * math.sqrt(1 / fan_in)
                    nn.init.uniform_(self.params[block_id], -bound, bound)

                elif self.pattern[(in_block, out_block)] == "orthogonal":
                    max_channels = self.max_channels[block_id]

                    for axis in range(2):
                        if max(self.kernel_size_per_stride) == 1:
                            if axis == 1: break
                            axis_id = block_id
                        else: # if max(self.kernel_size_per_stride) == 1:
                            axis_id = "%s_d%d" % (block_id, axis)

                        if init in ["identical", "reverse", "permutation"]:
                            matrix = torch.eye(max_channels)
                            if init == "reverse" and axis == 0:
                                matrix = matrix.flip(1)
                            elif init == "permutation":
                                matrix = matrix[:, torch.randperm(max_channels)]
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

    def paraunitary(self, in_block, out_block):
        """
        Construction of the paraunitary matrix for the given group.

        Argument:
        ---------
        in_block, out_block: int
            The input and output block for the paraunitary matrix.
            Note: the block identity has form "g%d-%d" % (in_block, out_block).

        Return:
        ------- 
        kernel: a 4th-order tensor of size 
            [out_channels, in_channels, kernel_size[0], kernel_size[1]]
            The paraunitary matrix for the given group.

        """
        block_id = "g%d-%d" % (in_block, out_block)
        max_channels = self.max_channels[block_id]

        if max(self.kernel_size_per_stride) == 1:

            # [max_channels, max_channels, *kernel_size_per_stride]
            kernel = self.module[block_id].matrix.unsqueeze(-1).unsqueeze(-1)

        else: # if max(self.kernel_size_per_stride) == 1:
            factors = [None] * 2
            for axis in range(2):
                axis_id = "%s_d%d" % (block_id, axis)

                # [1, out_channels_, in_channels_]
                factor = self.module[axis_id].matrix.unsqueeze(0)
                zeros  = torch.zeros_like(factor)

                # [kernel_size_per_stride, max_channels, max_channels]
                for pos in range(2, self.kernel_size_per_stride[axis] + 1):
                    factor_padded_left  = torch.cat((zeros, factor), dim = 0)
                    factor_padded_right = torch.cat((factor, zeros), dim = 0)

                    if pos % 2 == 0:
                        matrix = self.module["%s_r%d" % (axis_id, pos // 2)].matrix
                        factor = factor_padded_right + torch.einsum("kts,sr,nr->ktn", 
                            factor_padded_left - factor_padded_right, matrix, matrix)
                    else: # if pos % 2 == 1:
                        matrix = self.module["%s_l%d" % (axis_id, pos // 2)].matrix
                        factor = factor_padded_left  + torch.einsum("kts,tr,nr->kns",
                            factor_padded_right - factor_padded_left, matrix, matrix)

                factors[axis] = factor

            # [max_channels, max_channels, *kernel_size_per_stride]
            kernel = torch.einsum("hrs,wtr->tshw", factors)

        # [out_channels, in_channels_, *kernel_size_per_stride]
        kernel = kernel[:self.out_channels[out_block], :self.in_channels_[in_block]]

        if max(self.stride) > 1:
            # [out_channels, in_channels, *stride, *kernel_size_per_stride]
            kernel = kernel.view(self.out_channels[out_block], self.in_channels[in_block],
                self.stride[0], self.stride[1], self.kernel_size_per_stride[0], self.kernel_size_per_stride[1])

            # [out_channels, in_channels, *kernel_size]
            kernel = kernel.permute(0, 1, 4, 2, 5, 3).contiguous()
            kernel = kernel.view(self.out_channels[out_block], 
                self.in_channels[in_block], self.kernel_size[0], self.kernel_size[1])

        return kernel

    def weights(self):
        """
        Construction of the weights for forward computation.

        Return:
        -------
        weights: a 4th-order tensor of size
            [total_out_channels, total_in_channels, kernel_size[0], kernel_size[1]]
            The convolutional kernel for forward computation.

        """
        weights = [None] * len(self.in_channels)
        for in_block in range(len(self.in_channels)):

            kernels = [None] * len(self.out_channels)
            for out_block in range(len(self.out_channels)):
                block_id = "g%d-%d" % (in_block, out_block)

                if self.pattern[(in_block, out_block)] != "orthogonal":
                    kernels[out_block] = self.params[block_id]
                else: # if self.pattern[(in_block, out_block)] == "orthogonal":
                    kernels[out_block] = self.paraunitary(in_block, out_block)

            weights[in_block] = torch.cat(kernels, dim = 0)

        weights = torch.cat(weights, dim = 1)

        return weights

    def forward(self, inputs, cache = False):
        """
        Computation of the block 2D-convolutional layer. 

        Argument:
        ---------
        inputs: a 4th-order tensor of size 
            [batch_size,  total_in_channels,  in_height,  in_width]
            The input to the orthogonal 2D-convolutional layer.

        cache: bool
            Whether to use the cached weights for computation.
            Note: Using cached weights can reduce computation in RNNs.
            Default: False

        Return:
        -------
        outputs: a 4th-order tensor of size 
            [batch_size, total_out_channels, out_length, out_width]
            The output of the orthgonal 2D-convolutional layer.

        Notes:
        ------
        If padding_mode == "circular":
            out_height = in_height / stride[0]
            out_width  = in_height / stride[1]
        If padding_mode == "zeros":
            out_height = (in_height + dilation[0] * (kernel_size[0] - 1) / stride[0]
            out_width  = (in_width  + dilation[1] * (kernel_size[1] - 1) / stride[1]

        """
        if not cache or self.kernel is None:
            self.kernel = self.weights()

        outputs = self.conv(inputs, self.kernel, self.bias)

        return outputs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # block 2D-convolutional layer
    print("Testing block 2D-convolutional layer")
    
    # batch size and input dimensions
    batch_size, in_height, in_width = 2, 60, 60

    # dilated or strided convolutions
    for (dilation, stride) in [((1, 1), (1, 1)), ((2, 3), (1, 1)), ((3, 2), (1, 1)), ((1, 1), (3, 2)), ((1, 1), (3, 2))]:

        # exact orthogonal, row-orthogonal, and column-orthogonal
        for (in_channels, out_channels_per_stride) in [([32, 24], [32, 24]), ([32, 24], [24, 24]), ([32, 24], [24, 24])]:
            out_channels = [out_channels_per_stride[l] * stride[0] * stride[1] for l in range(len(in_channels))]

            in_channels_  = sum(in_channels)
            out_channels_ = sum(out_channels)

            # effective kernel size for each polyphase 
            for kernel_size_per_stride in [(1, 1), (1, 3), (3, 1), (3, 3)]:
                kernel_size = (kernel_size_per_stride[0] * stride[0], kernel_size_per_stride[1] * stride[1])

                # print hyperparamters for the block 
                print("out_channels = %d, kernel_size = %s, stride = %s, dilation = %s" 
                    % (out_channels_,     kernel_size,      stride,      dilation     ))

                # compute output height/width according to padding mode
                out_height = in_height // stride[0]
                out_width  = in_width  // stride[1]

                # initialize the module for evaluation
                if max(stride) == 1 and max(dilation) == 1 and in_channels[1] == out_channels[1]:
                    layer = BlockConv2d(in_channels, out_channels, kernel_size, bias = False,
                        stride = stride, dilation = dilation, padding_mode = "circular", padding = "auto",
                        pattern = {(0, 0): "orthogonal", (0, 1): "zero", (1, 0): "zero", (1, 1): "identical"})
                else:
                    layer = BlockConv2d(in_channels, out_channels, kernel_size, bias = False,
                        stride = stride, dilation = dilation, padding_mode = "circular", padding = "auto",
                        pattern = {(0, 0): "orthogonal", (0, 1): "zero", (1, 0): "zero", (1, 1): "orthogonal"})

                # evaluate forward computation with randomized inputs
                inputs  = torch.randn(batch_size, in_channels_, in_height, in_width, requires_grad = True).to(device)
                outputs = layer(inputs)

                output_size = outputs.size()
                print("output size: ", output_size)
                assert  output_size[0] == batch_size and output_size[1] == out_channels_ \
                    and output_size[2] == out_height and output_size[3] == out_width

                # evaluate backward computation with randomized gradients
                grad_outputs = torch.randn(batch_size, out_channels_, out_height, out_width).to(device)
                grad_inputs  = torch.autograd.grad(outputs, inputs, grad_outputs)[0]
                
                # split the inputs and outputs into different blocks
                inputs  = torch.split(inputs,   in_channels, dim = 1)
                outputs = torch.split(outputs, out_channels, dim = 1)

                grad_outputs = torch.split(grad_outputs, out_channels, dim = 1)
                grad_inputs  = torch.split(grad_inputs,   in_channels, dim = 1)

                # check orthogonal convolution
                if out_channels[0] >= in_channels[0] * stride[0] * stride[1]: # column-orthogonal
                    norm_inputs  = torch.norm(inputs[0]) 
                    norm_outputs = torch.norm(outputs[0])

                    if not torch.isclose(norm_inputs, norm_outputs, rtol = 1e-3, atol = 1e-4):
                        print("norm_inputs: %.4f, norm_outputs: %.4f" % 
                            (norm_inputs.item(), norm_outputs.item()))
                        print("The input norm and output norm do not match.")

                if out_channels[0] <= in_channels[0] * stride[0] * stride[0]: # row-orthogonal
                    norm_grad_inputs  = torch.norm(grad_inputs[0])
                    norm_grad_outputs = torch.norm(grad_outputs[0])

                    if not torch.isclose(norm_grad_outputs, norm_grad_inputs, rtol = 1e-3, atol = 1e-4):
                        print("norm_grad_inputs: %.4f, norm_grad_outputs: %.4f" % 
                            (norm_grad_inputs.item(), norm_grad_outputs.item()))
                        print("The input gradient norm and output gradient norm do not match.")

                print('-' * 60)
