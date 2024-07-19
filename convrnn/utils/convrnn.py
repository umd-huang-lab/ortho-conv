# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.utils as utils

from activation  import GroupSort, SReLU, modReLU
from blockconv2d import BlockConv2d
from orthoconv2d import OrthoConv2d


## Convolutional-RNN Module
class ConvRNNCell(nn.Module):

    def __init__(self,
            input_channels,
            hidden_channels,
            kernel_size,
            ortho_states = True,
            ortho_inputs = False,
            ortho_init = "torus",
            gain = 1,
            norm = False,
            activation = "modrelu",
            bias = True,
        ):
        """
        Construction of a Convolutional-RNN cell.
        
        Arguments:
        ----------
        [hyperparameters for the input/output channels]
        input_channels: int
            The number of input channels to the Conv-RNN cell.
        hidden_channels: int
            The number of hidden/output channels of the Conv-RNN cell.

        [hyperparameters for the filter bank]
        kernel_size: int or tuple(int, int)
            The size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k).
        
        # orthogonal convolutional layer
        ortho_states: bool
            Whether to use orthogonal convolution in states-states transition.
            Default: True
        ortho_inputs: bool
            Whether to use orthogonal convolution in inputs-states transition.
            Default: False
        ortho_init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of orthogonal convolutional layers.
            Default: "torus"
        
        # normal convolutional layer
        gain: float
            The scaling factor of the normal convolutional layer.
            Default: 1
        norm: bool
            Whether to apply normalization after the inputs-states transition.
            Default: False

        [hyperparameters for the nonlinear activation]
        activation: str 
            The activation function of the Conv-RNN cell.
            Options: "sigmoid", "tanh", "relu", "modrelu", "groupsort"
            Default: "modrelu"
        bias: bool
            Whether to add a bias before the activation function.
            default: True

        """
        super(ConvRNNCell, self).__init__()

        # hyperparameters of the input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._pair(kernel_size)

        # Note: the hidden states are not intialized in construction
        self.hidden_states = None

        # initialization of the convolutional layer
        self.conv = BlockConv2d([hidden_channels, input_channels], [hidden_channels], kernel_size,
            pattern = {(0, 0): "orthogonal" if ortho_states else "normal",
                       (0, 1): "orthogonal" if ortho_inputs else "normal"},
                       init = ortho_init, gain = gain, bias = bias)

        # initialization of the activation layer
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "srelu":
            self.act = SReLU(num_features = hidden_channels)
        elif activation == "modrelu":
            self.act = modReLU(num_features = hidden_channels)
        elif activation == "groupsort":
            self.act = GroupSort()
        else:
            raise NotImplementedError

        if norm:
             # initialization of the normalization layer
            self.norm = nn.GroupNorm(input_channels, input_channels)

            self.func = lambda inputs, states, first_step: self.act(self.conv(
                torch.cat([states, self.norm(inputs)], dim = 1), cache = not first_step))

        else: # if not norm:
            self.func = lambda inputs, states, first_step: self.act(self.conv(
                torch.cat([states, inputs], dim = 1), cache = not first_step))

    def init_states(self, inputs):
        """
        Initialization of the states in the Conv-RNN cell.
        
        Argument: 
        ---------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, input_height, input_width]
            The inputs to the Conv-RNN cell.

        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize hidden states to all zeros
        self.hidden_states = torch.zeros(batch_size,
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of convolutional-RNN cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width] 
            The inputs to the Conv-RNN cell.

        first_step: bool
            Whether the current pass is the first step in the input sequence. 
            Note: If so, hidden states are intialized to zeros tensors.
            default: False
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size 
            [batch_size, hidden_channels, height, width]
            The hidden states (and outputs) of the Conv-RNN cell.

        """
        if first_step: self.init_states(inputs)

        self.hidden_states = self.func(inputs, self.hidden_states, first_step)

        return self.hidden_states 


## Convolutional-GRU Module
class ConvGRUCell(nn.Module):

    def __init__(self,
            input_channels,
            hidden_channels,
            kernel_size,
            ortho_states = True,
            ortho_inputs = False,
            ortho_init = "torus",
            gain = 1,
            norm = False,
            activation = "modrelu", 
            bias = True,
        ):
        """
        Construction of convolutional-GRU cell.
        
        Arguments:
        ----------
        [hyperparameters for the input/output channels]
        input_channels: int
            The number of input channels to the Conv-GRU cell.
        hidden_channels: int
            The number of hidden/output channels of the Conv-GRU cell.

        [hyperparameters for the filters]
        kernel_size: int or tuple(int, int)
            The size of the (squared) convolutional kernel.
            Note: If the size is a single scalar k, it will be mapped to (k, k).

        # orthogonal convolutional layer
        ortho_states: bool
            Whether to use orthogonal convolution in states-states transition.
            Default: True
        ortho_inputs: bool
            Whether to use orthogonal convolution in inputs-states transition.
            Default: False
        ortho_init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of orthogonal convolutional layers.
            Default: "torus"
        
        # normal convolutional layer
        gain: float
            The scaling factor of the normal convolutional layer.
            Default: 1
        norm: bool
            Whether to apply normalization after the inputs-states transition.
            Default: False

        [hyperparameters for the nonlinear activation]
        activation: str
            The activation function of the Conv-GRU cell.
            default: "modrelu"
        bias: bool
            Whether to add a bias before the activation function.
            default: True

        """
        super(ConvGRUCell, self).__init__()

        # hyperparameters of the input/output interfaces
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        kernel_size = utils._pair(kernel_size)

        # Note: the hidden states are not intialized in construction
        self.hidden_states = None

        # initialization of the activation function
        if activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "srelu":
            self.act = SReLU(num_features = hidden_channels)
        elif activation == "modrelu":
            self.act = modReLU(num_features = hidden_channels)
        elif activation == "groupsort":
            self.act = GroupSort()
        else:
            raise NotImplementedError

        # initialization of the convolutional layers
        self.conv_g = nn.Conv2d(in_channels = input_channels + hidden_channels, 
            out_channels = 2 * hidden_channels, kernel_size = kernel_size,
            padding = (kernel_size[0] // 2, kernel_size[1] // 2), bias = bias)

        self.gates = lambda inputs, hidden_states: torch.split(
            torch.sigmoid(self.conv_g(torch.cat([inputs, hidden_states], dim = 1))), hidden_channels, dim = 1)

        if ortho_states:
            self.conv_h = OrthoConv2d(hidden_channels, hidden_channels, kernel_size, init = ortho_init)
            self.func_h = lambda states, first_step: self.conv_h(states, cache = not first_step)
        else: # if ortho_states:
            self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size,
                padding = (kernel_size[0] // 2, kernel_size[1] // 2), bias = bias)
            self.func_h = lambda states, first_step: self.conv_h(states)

        if ortho_inputs:
            self.conv_x = OrthoConv2d(input_channels, hidden_channels, kernel_size, init = ortho_init)
            self.func_x = lambda inputs, first_step: self.conv_x(inputs, cache = not first_step)
        else: # if not ortho_inputs:
            self.conv_x = nn.Conv2d(input_channels, hidden_channels, kernel_size,
                padding = (kernel_size[0] // 2, kernel_size[1] // 2), bias = bias)
            self.func_x = lambda inputs, first_step: self.conv_x(inputs)

            bound = gain * math.sqrt(1 /(kernel_size[0] * kernel_size[1] * input_channels))
            nn.init.uniform_(self.conv_x.weight, -bound, bound)

        if not norm:
            # initialization of the normalization layer
            self.norm = nn.GroupNorm(input_channels, input_channels)

            self.func = lambda inputs, states, gates, first_step: self.act(
                self.func_x(self.norm(inputs), first_step) + gates * self.func_h(states, first_step))

        else: # if norm:
            self.func = lambda inputs, states, gates, first_step: self.act(
                self.func_x(inputs, first_step) + gates * self.func_h(inputs, first_step))

    def init_states(self, inputs):
        """
        Initialization of the states in the Conv-GRU cell.
        
        Argument: 
        ---------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, input_height, input_width]
            The inputs to the Conv-GRU cell.

        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize hidden states to all zeros
        self.hidden_states = torch.zeros(batch_size,
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False, checkpointing = False):
        """
        Computation of convolutional-GRU cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width] 
            The inputs to the Conv-GRU cell.

        first_step: bool
            Whether the current pass is the first step in the input sequence. 
            Note: If so, hidden states are intialized to zeros tensors.
            default: False
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size 
            [batch_size, hidden_channels, height, width]
            The hidden states (and outputs) of the Conv-GRU cell.

        """
        if first_step: self.init_states(inputs)

        reset_gates, update_gates = self.gates(inputs, self.hidden_states)
        new_gates = self.func(inputs, self.hidden_states, reset_gates, first_step)

        self.hidden_states = update_gates * self.hidden_states + (1 - update_gates) * new_gates

        return self.hidden_states 


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # # Convolutional RNN cell
    # print("Testing Convolutional RNN cell...")
    # batch_size, in_features, num_features, height, width = 2, 32, 48, 64, 64

    # for ortho_states in [False, True]:
    #     for ortho_inputs, norm in [(False, True), (True, False)]:
    #         for activation in ["sigmoid", "tanh", "relu", "srelu", "modrelu", "groupsort"]:
    #             print("ortho_states = %s; ortho_inputs = %s; activation = %s" 
    #                 % (ortho_states,      ortho_inputs,      activation))

    #             inputs = torch.zeros(batch_size, in_features, height, width).to(device)
    #             module = ConvRNNCell(in_features, num_features, kernel_size = 5, 
    #                 ortho_states = ortho_states, ortho_inputs = ortho_inputs, 
    #                 activation = activation).to(device)

    #             for first_step in [True, False]:
    #                 outputs = module(inputs, first_step = first_step)

    #             print(outputs.size())
    #             print('-' * 60)

    # # Convolutional-GRU cell
    # print("Testing Convolutional-GRU cell...")
    # batch_size, in_features, num_features, height, width = 2, 32, 48, 64, 64

    # for ortho_states in [False, True]:
    #     for ortho_inputs, norm in [(False, True), (True, False)]:
    #         for activation in ["sigmoid", "tanh", "relu", "srelu", "modrelu", "groupsort"]:
    #             print("ortho_states = %s; ortho_inputs = %s; activation = %s" 
    #                 % (ortho_states,      ortho_inputs,      activation))

    #             inputs = torch.zeros(batch_size, in_features, height, width).to(device)
    #             module = ConvGRUCell(in_features, num_features, kernel_size = 5, 
    #                 ortho_states = ortho_states, ortho_inputs = ortho_inputs, 
    #                 activation = activation).to(device)

    #             for first_step in [True, False]:
    #                 outputs = module(inputs, first_step = first_step)

    #             print(outputs.size())
    #             print('-' * 60)
