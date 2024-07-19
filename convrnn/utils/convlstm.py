# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

from torch.utils.checkpoint import checkpoint


## Utilities
@torch.jit.script
def fuse_mul_add_mul(f, cell_states, i, g):
    return f * cell_states + i * g

def chkpt_blk(cc_i, cc_f, cc_o, cc_g, cell_states):
    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.tanh(cc_g)
    
    cell_states = fuse_mul_add_mul(f, cell_states, i, g)
    outputs = o * torch.tanh(cell_states)

    return outputs, cell_states

def factorize(number, order):
    d = {
        (1,  2): [1,  1], (1,  3): [1, 1, 1],
        (3,  2): [1,  3], (3,  3): [1, 1, 3],
        (32, 2): [4,  8], (32, 3): [2, 4, 4],
        (48, 2): [6,  8], (48, 3): [3, 4, 4],
        (64, 2): [8,  8], (64, 3): [4, 4, 4],
        (80, 2): [8, 10], (80, 3): [4, 4, 5]
    }
    return d[(number, order)]


## Standard Convolutional-LSTM Module
class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size = 5, bias = True):
        """
        Construction of convolutional-LSTM cell.
        
        Arguments:
        ----------
        (Hyper-parameters of input/output interfaces)
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.

        (Hyper-parameters of the convolutional opeations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 3
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True

        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(
            in_channels  = input_channels + hidden_channels, 
            out_channels = 4 * hidden_channels,
            kernel_size = kernel_size, padding = padding, bias = bias)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None

    def initialize(self, inputs):
        """
        Initialization of convolutional-LSTM cell.
        
        Arguments: 
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, input_height, input_width]
            Input tensor of convolutional-LSTM cell.

        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)
        self.cell_states   = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of convolutional-LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            Note: If so, both hidden and cell states are intialized to zeros tensors.
            default: False

        checkpointing: bool
            Whether to use the checkpointing technique to reduce memory expense.
            default: True
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size 
            [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.

        """
        if first_step: self.initialize(inputs)

        concat_conv = self.conv(torch.cat([inputs, self.hidden_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        if checkpointing:
            self.hidden_states, self.cell_states = checkpoint(chkpt_blk, cc_i, cc_f, cc_o, cc_g, self.cell_states)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
    
            self.cell_states = fuse_mul_add_mul(f, self.cell_states, i, g)
            self.hidden_states = o * torch.tanh(self.cell_states)
        
        return self.hidden_states 


## Convolutional Tensor-Train LSTM Module
class ConvTTLSTMCell(nn.Module):

    def __init__(self,
        # interface of the Conv-TT-LSTM 
        input_channels, hidden_channels,
        # convolutional tensor-train network
        order = 3, steps = 3, ranks = 8,
        # convolutional operations
        kernel_size = 5, bias = True):
        """
        Initialization of convolutional tensor-train LSTM cell.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        input_channels:  int
            Number of input channels of the input tensor.
        hidden_channels: int
            Number of hidden/output channels of the output tensor.
        Note: the number of hidden_channels is typically equal to the one of input_channels.

        (Hyper-parameters of the convolutional tensor-train format)
        order: int
            The order of convolutional tensor-train format (i.e. the number of core tensors).
            default: 3
        steps: int
            The total number of past steps used to compute the next step.
            default: 3
        ranks: int
            The ranks of convolutional tensor-train format (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 5
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True
        """
        super(ConvTTLSTMCell, self).__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        ## Convolutional tensor-train network
        self.steps = steps
        self.order = order
        self.lags  = steps - order + 1

        ## Convolutional operations
        kernel_size = utils._pair(kernel_size)
        padding     = kernel_size[0] // 2, kernel_size[1] // 2

        Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size, padding = padding, bias = bias)

        ## Convolutional layers
        self.layers  = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        for l in range(order):
            self.layers.append(Conv2d(
                in_channels  = ranks if l < order - 1 else ranks + input_channels, 
                out_channels = ranks if l < order - 1 else 4 * hidden_channels))

            self.layers_.append(Conv2d(
                in_channels = self.lags * hidden_channels, out_channels = ranks))

    def initialize(self, inputs):
        """ 
        Initialization of the hidden/cell states of the convolutional tensor-train cell.

        Arguments:
        ----------
        inputs: 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Input tensor to the convolutional tensor-train LSTM cell.

        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states  = [torch.zeros(batch_size, self.hidden_channels, 
            height, width, device = device) for t in range(self.steps)]
        self.hidden_pointer = 0 # pointing to the position to be updated

        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of the convolutional tensor-train LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the tensor is the first step in the input sequence. 
            Note: If so, both hidden and cell states are intialized to zeros tensors.
            default: False

        checkpointing: bool
            Whether to use the checkpointing technique to reduce memory expense.
            default: True
        
        Returns:
        --------
        hidden_states: a list of 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.

        """

        if first_step: self.initialize(inputs) # intialize the states at the first step

        ## (1) Convolutional tensor-train module
        for l in range(self.order):
            input_pointer = self.hidden_pointer if l == 0 else (input_pointer + 1) % self.steps

            input_states = self.hidden_states[input_pointer:] + self.hidden_states[:input_pointer]
            input_states = input_states[:self.lags]

            input_states = torch.cat(input_states, dim = 1)
            input_states = self.layers_[l](input_states)

            if l == 0:
                temp_states = input_states
            else: # if l > 0:
                temp_states = input_states + self.layers[l-1](temp_states)
                
        ## (2) Standard convolutional-LSTM module
        concat_conv = self.layers[-1](torch.cat([inputs, temp_states], dim = 1))
        cc_i, cc_f, cc_o, cc_g = torch.split(concat_conv, self.hidden_channels, dim = 1)

        if checkpointing:
            outputs, self.cell_states = checkpoint(chkpt_blk, cc_i, cc_f, cc_o, cc_g, self.cell_states)
        else:
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
    
            self.cell_states = fuse_mul_add_mul(f, self.cell_states, i, g)
            outputs = o * torch.tanh(self.cell_states)

        self.hidden_states[self.hidden_pointer] = outputs
        self.hidden_pointer = (self.hidden_pointer + 1) % self.steps
        
        return outputs


## Tensor-Train Convolutional Tensor-Train LSTM Module
class TTConvLSTMCell(nn.Module):

    def __init__(self,
        # interface of the TT-ConvLSTM 
        input_channels, hidden_channels,
        # tensor-train decomposition
        order = 3, ranks = 8,
        # convolutional operations
        kernel_size = 5, bias = True):
        """
        Initialization of Tensor-Train Convolutional-LSTM (TT-ConvLSTM) cell.

        Arguments:
        ----------
        (Hyper-parameters of the input/output channels)
        input_channels:  int
            Number of input channels of the input tensor.
        hidden_channels: int
            Number of hidden/output channels of the output tensor.
        Note: the number of hidden_channels is typically equal to the one of input_channels.

        (Hyper-parameters of the tensor-train format)
        order: int
            The order of tensor-train decomposition (i.e. the number of core tensors).
            default: 3
        ranks: int
            The ranks of tensor-train decomposition (where all ranks are assumed to be the same).
            default: 8

        (Hyper-parameters of the convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k)
            default: 5
        bias: bool
            Whether or not to add the bias in each convolutional operation.
            default: True

        """
        super(TTConvLSTMCell, self).__init__()

        ## Input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        ## Tensor-Train decomposition
        self.order = order
        self.ranks = ranks

        ## Channels factorization
        self.input_shape  = factorize( input_channels, order)
        self.hidden_shape = factorize(hidden_channels, order)

        ## Convolutional operations
        self.kernel_size = utils._pair(kernel_size)
        padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

        self.conv_h = nn.Conv2d(
            in_channels = hidden_channels, out_channels = 4 * hidden_channels,
            kernel_size = kernel_size, padding = padding, bias = bias)

        self.conv_x = lambda inputs, weights: \
            F.conv2d(inputs, weights, None, padding = padding)

        self.kernels = nn.ParameterDict()
        for gate in ["i", "f", "g", "o"]:
            self.kernels["%s_%d" % (gate, 0)] = nn.Parameter(
                torch.Tensor(self.input_shape[0], self.hidden_shape[0], ranks))
  
            for l in range(1, order):
                self.kernels["%s_%d" % (gate, l)] = nn.Parameter(
                    torch.Tensor(self.input_shape[l], self.hidden_shape[l], ranks, ranks))

            self.kernels["%s_%d" % (gate, order)] = nn.Parameter(
                torch.Tensor(self.kernel_size[0], self.kernel_size[1], ranks))

        self.init_params()

        equation = []
        i, o, r = ord("a"), ord("i"), ord("r")

        equation.append("".join([chr(o), chr(i), chr(r), ","]))
        i, o, r = i + 1, o + 1, r + 1

        for l in range(1, order):
            equation.append("".join([chr(o), chr(i), chr(r-1), chr(r), ","]))
            i, o, r = i + 1, o + 1, r + 1

        equation.append("".join(["h", "w", chr(r-1), "->"]))

        equation.append("".join([chr(l) for l in range(ord("i"), ord("i") + order)] + 
            [chr(l) for l in range(ord("a"), ord("a") + order)] + ["h", "w"]))

        self.equation = "".join(equation)

    def init_params(self):
        """
        Initialization of the Tensor-Train kernels.

        """
        for gate in ["i", "f", "g", "o"]:
            fan_in, fan_out = self.input_shape[0], self.hidden_shape[0] * self.ranks
            bound = math.sqrt(6./(fan_in + fan_out))
            torch.nn.init.uniform_(self.kernels["%s_0" % gate], -bound, bound)

            for l in range(1, self.order):
                fan_in, fan_out = self.input_shape[l] * self.ranks, self.hidden_shape[l] * self.ranks
                bound = math.sqrt(6./(fan_in + fan_out))
                torch.nn.init.uniform_(self.kernels["%s_%d" % (gate, l)], -bound, bound)

            fan_in  = self.kernel_size[0] * self.kernel_size[1] * self.ranks
            fan_out = self.kernel_size[0] * self.kernel_size[1] * 1
            bound = math.sqrt(6./(fan_in + fan_out))
            torch.nn.init.uniform_(self.kernels["%s_%d" % (gate, self.order)], -bound, bound)

    def init_states(self, inputs):
        """ 
        Initialization of the hidden/cell states.

        Arguments:
        ----------
        inputs: 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Input tensor to the Tensor-Train Convolutional-LSTM cell.

        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)
        self.cell_states   = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of the Tensor-Train Convolutional-LSTM (TT-ConvLSTM) cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Input to the TT-ConvLSTM cell.

        first_step: bool
            Whether the input is the first step in the sequence. 
            Note: If so, both hidden and cell states are intialized to zeros tensors.
            default: False

        checkpointing: bool
            Whether to use the checkpointing technique to reduce memory expense.
            default: True
        
        Returns:
        --------
        hidden_states: a list of 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Hidden states (and outputs) of the TT-ConvLSTM cell.

        """
        if first_step: self.init_states(inputs)

        cc = {}
        cc["i"], cc["f"], cc["o"], cc["g"] = torch.split(
            self.conv_h(self.hidden_states), self.hidden_channels, dim = 1)

        for gate in ["i", "f", "o", "g"]:
            weights = torch.einsum(self.equation, [self.kernels["%s_%d" % (gate, l)] for l in range(self.order + 1)])
            weights = weights.view(self.hidden_channels, self.input_channels, self.kernel_size[0], self.kernel_size[1])
            cc[gate] += self.conv_x(inputs, weights)

        if checkpointing:
            self.hidden_states, self.cell_states = checkpoint(chkpt_blk, cc["i"], cc["f"], cc["o"], cc["g"], self.cell_states)
        else:
            i = torch.sigmoid(cc["i"])
            f = torch.sigmoid(cc["f"])
            o = torch.sigmoid(cc["o"])
            g = torch.tanh(   cc["g"])
    
            self.cell_states = fuse_mul_add_mul(f, self.cell_states, i, g)
            self.hidden_states = o * torch.tanh(self.cell_states)
        
        return self.hidden_states


if __name__ == '__main__':
    device = "cuda:0"
    inputs = torch.zeros(2, 48, 64, 64).to(device)
    cell = TTConvLSTMCell(48, 80, order = 3).to(device)
    outputs = cell(inputs, True)
    outputs = cell(inputs, False)
    print(outputs.size())
