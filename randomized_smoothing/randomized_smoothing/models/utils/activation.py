import torch
import torch.nn as nn
import torch.nn.functional as F


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


class modReLU(nn.Module):
    def __init__(self, num_features, num_dims = 2, init = 0.01):
        """
        Construction of a modReLU activation.

        Arguments:
        ----------
        num_features: int
            The number of input/output features.
            
        num_dims: int
            The number of dimensions in each feature.
            Default: 2

        init: float
            The initial range of the bias vector.
            Note: each scalar is initialized with U[-init, init].
            Default: 0.01
        
        """
        super(modReLU, self).__init__()

        self.bias = nn.Parameter(torch.zeros([num_features] + [1] * num_dims))
        self.bias.data.uniform_(-init, init)

    def forward(self, inputs):
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

        # amplitude and phase of the input tensor
        norm, phase = torch.abs(inputs), torch.sign(inputs)

        # amplitude of the output tensor
        magnitude = F.relu(norm + self.bias, True)

        # construct the output tensor
        outputs = magnitude * phase 

        return outputs


class SReLU(nn.Module):

    def __init__(self, num_features, num_dims = 2, init = -1.0):
        """
        Construction of a shifted ReLU activation.

        Arguments:
        ----------
        num_features: int
            The number of input/output features.
            
        num_dims: int
            The number of dimensions in each feature.
            Default: 2

        init: float
            The initial range of the bias vector.
            Note: each scalar is initialized with U[-init, init].
            Default: 0.01

        """
        super(SReLU, self).__init__()

        self.bias = nn.Parameter(torch.zeros([num_features] + [1] * num_dims))
        nn.init.constant_(self.bias, -1.0)

    def forward(self, inputs):
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
        outputs = F.relu(inputs - self.bias, False) + self.bias

        return outputs


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # modReLU activation
    print("Testing modReLU activation...")
    batch_size, num_features, feature_size = 2, 64, 32

    for num_dims in [0, 1, 2, 3]:
        inputs = torch.randn([batch_size, num_features] + [feature_size] * num_dims)
        module = modReLU(num_features = num_features, num_dims = num_dims)

        outputs = module(inputs)
        print("output size: ", outputs.size())


    # shfited ReLU activation
    print("Testing shifted ReLU activation...")
    batch_size, num_features, feature_size = 2, 64, 32

    for num_dims in [0, 1, 2, 3]:
        inputs = torch.randn([batch_size, num_features] + [feature_size] * num_dims)
        module = SReLU(num_features = num_features, num_dims = num_dims)

        outputs = module(inputs)
        print("output size: ", outputs.size())


    # GroupSort activation
    print("Testing GroupSort activation...")
    batch_size, num_features, feature_size = 2, 64, 32

    for group_size in [2, 4, num_features]:
        for num_dims in [0, 1, 2, 3]:
            inputs = torch.randn([batch_size, num_features] + [feature_size] * num_dims)
            module = GroupSort(group_size = group_size, axis = 1)

            outputs = module(inputs)
            print("output size: ", outputs.size())
