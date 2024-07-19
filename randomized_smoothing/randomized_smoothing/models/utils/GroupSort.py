import sys
import warnings

import math
import torch
import torch.nn as nn

from torch.autograd import Function


class GroupSort(nn.Module):
    def __init__(self, group_size = 2, axis = 1):
        """
        Construction of a GroupSort activation.
        
        Arguments:
        ----------
        group_size: int
            The grouping size of the features.
            Default: 2

        axis: int
            The axis for indexing the features.
            Default: 1

        """
        super(GroupSort, self).__init__()

        self.group_size = group_size
        self.axis = axis

    def forward(self, inputs):
        """
        Computation of the GroupSort activation.

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
        shape = list(inputs.shape)
        num_features = shape[self.axis]
        assert num_features % self.group_size == 0

        if self.group_size == 2:
            a, b = inputs.split(inputs.size(self.axis) // 2, self.axis)
            a, b = torch.max(a, b),  torch.min(a, b)
            return torch.cat([a, b], dim = self.axis)

        else: # if self.group > 2:
            shape[self.axis] = num_features // self.group_size
            shape.insert(self.axis, self.group_size)
            return inputs.view(*shape).sort(dim = self.axis)[0].view(*inputs.shape)

if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    ## Test the max_min function


    ## Test the group_sort function


    ## Test the group sorting layer
    print("Testing the group sorting layer...")
    
    # batch size
    batch_size = 2

    print('-' * 60)