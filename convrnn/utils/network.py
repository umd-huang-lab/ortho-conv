# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

import torch
import torch.nn as nn

from convrnn  import ConvRNNCell, ConvGRUCell
from convlstm import ConvLSTMCell, ConvTTLSTMCell, TTConvLSTMCell


## Convolutional-RNN network
class ConvLSTMNet(nn.Module):
    def __init__(self,
        # input to the model
        input_channels,
        # architecture of the model
        layers_per_block, hidden_channels, skip_stride = None,
        # parameters of convolutional tensor-train layers
        cell = "convlstm", cell_params = {}, 
        # parameters of convolutional operation
        kernel_size = 3, bias = True,
        # output function and output format
        output_sigmoid = False):
        """
        Initialization of a convolutional recurrent network.
        
        Arguments:
        ----------
        [hyper-parameters of the input/output interfaces]
        input_channels: int
            The number of channels of the input video frames.
            Note: 3 for colored video frames, 1 for gray video frames.

        output_sigmoid: bool
            Whether to apply sigmoid function after the output layer.
            Default: False

        [Hyper-parameters of the model architecture]
        layers_per_block: list of ints
            The number of recurrent layers in each block.

        hidden_channels: list of ints
            Number of output channels.
            Note: The length of hidden_channels is equal to the number of blocks.

        skip_stride: int
            The stride (in term of blocks) of the skip connections.
            Note: The skip_stride should be larger than the number of blocks.
            Default: None, i.e. no skip connection
        
        [hyper-parameters of the convolutional recurrent layers]
        cell: str
            The type of the convolutional convolutional recurrent layers.
            Options: "convlstm", "convttlstm", "ttconvlstm", "convrnn", or "convgru"
            Default: "convlstm"
        cell_params: dictionary
            The hyper-parameters of the convolutional recurrent layers.
            Default: {}

        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            default: 3
        bias: bool 
            Whether to add bias in the convolutional operation.
            default: True

        """
        super(ConvLSTMNet, self).__init__()

        ## Hyperparameters
        self.layers_per_block = layers_per_block
        self.hidden_channels  = hidden_channels

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), "Invalid number of blocks."

        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride

        self.output_sigmoid = output_sigmoid

        ## Module type of convolutional LSTM layers

        if cell == "convlstm": # standard convolutional LSTM
            Cell = lambda in_channels, out_channels: ConvLSTMCell(
                input_channels = in_channels, hidden_channels = out_channels,
                kernel_size = kernel_size, bias = bias,
            )

        elif cell == "convttlstm": # convolutional tensor-train LSTM
            Cell = lambda in_channels, out_channels: ConvTTLSTMCell(
                input_channels = in_channels, hidden_channels = out_channels,
                kernel_size = kernel_size, bias = bias,
                order = cell_params["order"],
                steps = cell_params["steps"],
                ranks = cell_params["ranks"], 
            )

        elif cell == "ttconvlstm": # tensor-train convolutional LSTM
            Cell = lambda in_channels, out_channels: TTConvLSTMCell(
                input_channels  = in_channels, hidden_channels = out_channels,
                kernel_size = kernel_size, bias = bias,
                order = cell_params["order"], 
                ranks = cell_params["ranks"],
            )

        elif cell == "convrnn": # convolutional RNN
            Cell = lambda in_channels, out_channels: ConvRNNCell(
                in_channels, out_channels, kernel_size, bias = bias,
                ortho_states = cell_params["ortho_states"],
                ortho_inputs = cell_params["ortho_inputs"],
                ortho_init = cell_params["ortho_init"],
                activation = cell_params["activation"],
                gain = cell_params["gain"],
                norm = cell_params["norm"],
            )

        elif cell == "convgru": # convolutional GRU
            Cell = lambda in_channels, out_channels: ConvGRUCell(
                input_channels = in_channels, hidden_channels = out_channels, 
                kernel_size = kernel_size, bias = bias,
                ortho_states = cell_params["ortho_states"],
                ortho_inputs = cell_params["ortho_inputs"],
                ortho_init = cell_params["ortho_init"],
                activation = cell_params["activation"],
                gain = cell_params["gain"],
                norm = cell_params["norm"],
            )

        else: # if cell not in ["convlstm", "convttlstm", "ttconvlstm", "convrnn", "convgru"]:
            raise NotImplementedError

        ## Construction of convolutional LSTM network

        # stack the convolutional-LSTM layers with skip connections 
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0: 
                    channels = hidden_channels[b]
                elif b == 0: # if l == 0 and b == 0:
                    channels = input_channels
                else: # if l == 0 and b > 0:
                    channels = hidden_channels[b-1]
                    if b > skip_stride:
                        channels += hidden_channels[b-1-skip_stride] 

                lid = "b{}l{}".format(b, l) # layer ID
                self.layers[lid] = Cell(channels, hidden_channels[b])

        # number of input channels to the last layer (output layer)
        channels = hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += hidden_channels[-1-self.skip_stride]

        self.layers["output"] = nn.Conv2d(channels, input_channels, 
            kernel_size = 1, padding = 0, bias = True)


    def forward(self, inputs, input_frames, future_frames, output_frames, 
        teacher_forcing = False, scheduled_sampling_ratio = 0, checkpointing = False):
        """
        Computation of the convolutional recurrent network.
        
        Arguments:
        ----------
        inputs: a 5-th order tensor of size
            [batch_size, input_frames, input_channels, height, width] 
            Input tensor (video) to the convolutional recurrent network.
        
        input_frames: int
            The number of input frames to the model.
        future_frames: int
            The number of future frames predicted by the model.
        output_frames: int
            The number of output frames returned by the model.

        teacher_forcing: bool
            Whether the model is trained in teacher_forcing mode.
            Note 1: In test mode, teacher_forcing should be set as False.
            Note 2: If teacher_forcing mode is on,  # of frames in inputs = total_steps
                    If teacher_forcing mode is off, # of frames in inputs = input_frames
        scheduled_sampling_ratio: float between [0, 1]
            The ratio of ground-truth frames used in teacher_forcing mode.
            default: 0 (i.e. no teacher forcing effectively)

        Returns:
        --------
        outputs: a 5-th order tensor of size
            [batch_size, output_frames, hidden_channels, height, width]
            Output tensor (video) of the convolutional recurrent network.

        """

        # compute the teacher forcing mask 
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            # generate the teacher_forcing mask (4-th order)
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio * 
                torch.ones(inputs.size(0), future_frames - 1, 1, 1, 1, device = inputs.device))
        else: # if not teacher_forcing or scheduled_sampling_ratio < 1e-6:
            teacher_forcing = False

        # the number of time steps in the computational graph
        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size [batch_size, input_channels, height, width]
            if t < input_frames: 
                input_ = inputs[:, t]
            elif not teacher_forcing:
                input_ = outputs[t-1]
            else: # if t >= input_frames and teacher_forcing:
                mask = teacher_forcing_mask[:, t - input_frames]
                input_ = inputs[:, t] * mask + outputs[t-1] * (1 - mask)

            queue = [] # previous outputs for skip connection
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l) # layer ID
                    input_ = self.layers[lid](input_, 
                        first_step = (t == 0), checkpointing = checkpointing)

                queue.append(input_)
                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue.pop(0)], dim = 1) # concat over the channels

            # map the hidden states to predictive frames (with optional sigmoid function)
            outputs[t] = self.layers["output"](input_)
            if self.output_sigmoid:
                outputs[t] = torch.sigmoid(outputs[t])

        # return the last output_frames of the outputs
        outputs = outputs[-output_frames:]

        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        outputs = torch.stack([outputs[t] for t in range(output_frames)], dim = 1)

        return outputs