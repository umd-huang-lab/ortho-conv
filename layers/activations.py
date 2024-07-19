import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple, Union


class SReLU(nn.Module):

    def __init__(
        self, 
        num_features: int,
        init: float = -1.0
    ) -> None:
        """
        Construction of a shifted ReLU activation.

        Arguments:
        ----------
        num_features: int
            The number of input/output features.

        init: float
            The initial values of the bias vector.
            Default: -1

        """
        super(SReLU, self).__init__()

        # initialize the bias vector
        self.bias = nn.Parameter(torch.Tensor(num_features))
        nn.init.constant_(self.bias, init)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the shifted ReLU activation.

        Argument:
        ---------
        inputs: a (d+2)th-order tensor of size
            [batch_size, num_features, feature_size_1, ..., feature_size_d]
            The inputs to the shifted ReLU activation.

        Return:
        -------
        outputs: a (d+2)th-order tensor of size
            [batch_size, num_features, feature_size_1, ..., feature_size_d]
            The outputs of the shifted ReLU activation.

        """
        bias = self.bias.view([1, -1] + [1] * (inputs.dim() - 2))

        outputs = F.relu(inputs - bias, False) + bias

        return outputs


class modReLU(nn.Module):

    def __init__(
        self, 
        num_features: int,
        num_dims: int = 2, 
        init: float = 0.01
    ) -> None:
        """
        Construction of a modReLU activation.

        Arguments:
        ----------
        num_features: int
            The number of input/output features.

        init: float
            The initial range of the bias vector.
            Note: each scalar is initialized with U[-init, init].
            Default: 0.01
        
        """
        super(modReLU, self).__init__()

        # initialize the bias vector
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.bias.data.uniform_(-init, init)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the modReLU activation.

        Argument:
        ---------
        inputs: a (d+2)th-order tensor of size
            [batch_size, num_features, feature_size_1, ..., feature_size_d]
            The inputs to the modReLU activation.

        Return:
        -------
        outputs: a (d+2)th-order tensor of size
            [batch_size, num_features, feature_size_1, ..., feature_size_d]
            The outputs of the modReLU activation.

        """
        bias = self.bias.view([1, -1] + [1] * (inputs.dim() - 2))

        # amplitude and phase of the input tensor
        norm, phase = torch.abs(inputs), torch.sign(inputs)

        # amplitude of the output tensor
        magnitude = F.relu(norm + bias, True)

        # construct the output tensor
        outputs = magnitude * phase 

        return outputs


class GroupSort(nn.Module):

    def __init__(
        self,
        group_size = 2
    ) -> None:
        """
        Construction of a GroupSort activation.
        
        Arguments:
        ----------
        group_size: int
            The grouping size of the features.
            Default: 2

        """
        super(GroupSort, self).__init__()

        self.group_size = group_size

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the GroupSort activation.

        Argument:
        ---------
        inputs: a (d+2)th-order tensor of size
            [batch_size, num_features, feature_size_1, ..., feature_size_d]
            The inputs to the GroupSort activation.

        Return:
        -------
        outputs: a (d+2)th-order tensor of size
            [batch_size, num_features, feature_size_1, ..., feature_size_d]
            The outputs of the GroupSort activation.

        """

        if self.group_size == 2:
            a, b = inputs.chunk(2, dim = 1)
            a, b = torch.max(a, b), torch.min(a, b)

            outputs = torch.cat([a, b], dim = 1)

        else: # if self.group > 2:
            shape = list(inputs.shape)

            shape[1] //= self.group_size
            shape.insert(1, self.group_size)

            outputs = inputs.view(*shape).sort(dim = 1)[0].view(*inputs.shape)

        return outputs


class Checkerboard(nn.Module):

    def __init__(self) -> None:
        """
        Construction of a 2D Checkerboard activation.

        """
        super(Checkerboard, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Computation of the GroupSort activation.

        Argument:
        ---------
        inputs: a 4th-order tensor of size
            [batch_size, num_features, feature_height, feature_width]
            The inputs to the 2D Checkerboard activation.

        Return:
        -------
        outputs: a 4th-order tensor of size
            [batch_size, num_features, feature_height, feature_width]
            The outputs of the 2D Checkerboard activation.

        """
        shape  = inputs.shape
        inputs = inputs.view(shape[0], shape[1], shape[2] // 2, 2, shape[3] // 2, 2)

        a, b = inputs.chunk(2, dim = 3)
        b = b.flip(dims = [5])
        inputs = torch.cat([a, b], dim = 3)

        inputs[..., 1] = -inputs[..., 1]

        a, b = inputs.chunk(2, dim = 1)
        a, b = torch.max(a, b), torch.min(a, b)
        outputs = torch.cat([a, b], dim = 1)

        outputs[..., 1] = -outputs[..., 1]

        a, b = outputs.chunk(2, dim = 3)
        b = b.flip(dims = [5])
        outputs = torch.cat([a, b], dim = 3)

        outputs = outputs.view(shape)

        return outputs


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Testing shifted ReLU activation...")
    batch_size, num_features, feature_size = 2, 32, 64

    for num_dims in [0, 1, 2, 3]:
        if num_dims > 0: 
            feature_size_per_dim = round(feature_size ** (1 / num_dims))
        else: # num_dims == 0: 
            feature_size_per_dim = None

        inputs = torch.randn([batch_size, num_features] + 
            [feature_size_per_dim] * num_dims).to(device)
        module = SReLU(num_features = num_features).to(device)

        outputs = module(inputs)
        print("output size: ", outputs.size())


    print("Testing modReLU activation...")
    batch_size, num_features, feature_size = 2, 32, 64

    for num_dims in [0, 1, 2, 3]:
        if num_dims > 0: 
            feature_size_per_dim = round(feature_size ** (1 / num_dims))
        else: # num_dims == 0: 
            feature_size_per_dim = None

        inputs = torch.randn([batch_size, num_features]
            + [feature_size_per_dim] * num_dims).to(device)
        module = modReLU(num_features = num_features).to(device)

        outputs = module(inputs)
        print("output size: ", outputs.size())


    print("Testing GroupSort activation...")
    batch_size, num_features, feature_size = 2, 32, 64

    for group_size in [2, 4, num_features]:
        for num_dims in [0, 1, 2, 3]:
            if num_dims > 0: 
                feature_size_per_dim = round(feature_size ** (1 / num_dims))
            else: # num_dims == 0: 
                feature_size_per_dim = None

            inputs = torch.randn([batch_size, num_features] +
                [feature_size_per_dim] * num_dims).to(device)
            module = GroupSort(group_size = group_size).to(device)

            outputs = module(inputs)
            print("output size: ", outputs.size())


    print("Testing Checkerboard activation...")
    batch_size, num_features, feature_height, feature_width = 2, 32, 8, 8

    inputs = torch.randn([batch_size, num_features, feature_height, feature_width]).to(device)
    module = Checkerboard()

    outputs = module(inputs)
    print("output size: ", outputs.size())
