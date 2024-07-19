import warnings

import torch
import geotorch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, List, Tuple, Union

from utilities import Matrix


class OrthoLinear(nn.Module):

    def __init__(
        self, 
        in_features:  int, 
        out_features: int,
        bias: bool = False, 
        triv: str = "expm",
        init: str = "uniform"
    ) -> None:
        """
        Construction of an orthogonal fully-connected layer.

        Arguments:
        ----------
        [hyperparameters for input/output features]
        in_features: int
            The number of input features to the orthogonal fully-connected layer.
        out_features: int
            The number of output features of the orthogonal fully-connected layer.
        Note: For orthogonality, in_features should equal to out_features.
            If in_features < out_features, 
                the layer is column-orthogonal (input norm perserving).
            If in_features > out_features, 
                the layer is row-orthogonal (gradient norm perserving).

        bias: bool
            Whether to add a learnable bias to the output.
            Note: For orthogonality, the learnable bias should be disabled.
            Default: False

        [hyperparameters for the trivialization and initialization]
        triv: str ("expm" or "cayley")
            The retraction that maps a skew-symmetric matrix to an orthogonal matrix.
            Default: "expm"
        init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of the orthogonal layer.
            Default: "uniform"

        """
        super(OrthoLinear, self).__init__()

        ## [hyperparameters for input/output features]
        self.in_features  =  in_features
        self.out_features = out_features

        if self.in_features > self.out_features:
            warnings.warn("The layer is made row-orthogonal.")
        elif self.in_features < self.out_features: 
            warnings.warn("The layer is made column-orthogonal.")

        ## [construction of the layer parameters]

        # parameter for the orthogonal matrix
        self.ortho = Matrix(self.in_features, self.out_features)
        geotorch.orthogonal(self.ortho, "matrix", triv = triv)

        self.kernel = None

        # parameter for the optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
            _bias = lambda outputs, vector: outputs - vector
        else: # if not bias:
            self.register_parameter('bias', None)
            _bias = lambda outputs, vector: outputs

        # initialize the layer parameters
        assert init in ["uniform", "identical", "reverse", "permutation", "torus"], \
            "The initialization method is not supported for orthogonal layer."

        self.reset_parameters(init = init)

        # [functionals for forward/reverse computations]
        self.linear_forward = lambda inputs,  weights, vector: \
            F.linear(inputs, weights, vector)
        self.linear_reverse = lambda outputs, weights, vector: \
            F.linear(_bias(outputs, vector), weights.transpose(0, 1))

    def reset_parameters(self, init: str) -> None:
        """
        Initialization of the orthogonal fully-connected layer.
        
        Argument:
        ---------
        init: str ("uniform", "torus", "identical", "reverse", or "permutation")
            The initialization of the orthogonal fully-connected layer.

        """

        # initialize the orthogonal matrix
        if init in ["identical", "reverse", "permutation"]:
            max_features = max(self.in_features, self.out_features)
            min_features = min(self.in_features, self.out_features)

            perm = torch.randperm(max_features)[:min_features]
            if init == "identical":
                perm, _ = torch.sort(perm, descending = False) 
            elif init == "reverse":
                perm, _ = torch.sort(perm, descending = True)

            matrix = torch.zeros(self.out_features, self.in_features)
            if self.out_features < self.in_features:
                matrix[:, perm] = torch.eye(min_features)
            else: # if self.out_features >= self.in_features:
                matrix[perm, :] = torch.eye(min_features)

        elif init == "torus":
            matrix = self.ortho.parametrizations.matrix[0].sample(init)

        if init != "uniform":
            self.ortho.matrix = matrix

        # initialize the bias to zeros for orthogonality
        if self.bias is not None:
            init.zeros_(self.bias)

    def weights(self) -> Tensor:
        """
        Construction of the weights for forward or reverse computation.

        Return:
        -------
        weights: a (2nd-order) matrix of size [out_features, in_features]
            The weights matrix for forward or reverse computation. 

        """
        return self.ortho.matrix

    def forward(self, inputs: Tensor, reverse: bool = False, cache: bool = False) -> Tensor:
        """
        Computation of the orthogonal fully-connected layer. 

        Argument:
        ---------
        inputs: a (2nd-order) matrix of size [batch_size, in_features]
            The input to the orthogonal fully-connected layer.

        Return:
        -------
        outputs: a (2nd-order) matrix of size [batch_size, out_features]
            The output of the orthgonal fully-connected layer.

        reverse: bool
            Whether to compute the layer's reverse pass.
            Default: False    

        cache: bool
            Whether to use the cached weights for computation.
            Default: False

        """
        if not cache or self.kernel is None:
            self.kernel = self.weights()

        if not reverse:
            outputs = self.linear_forward(inputs, self.kernel, self.bias)
        else: # if reverse:
            outputs = self.linear_reverse(inputs, self.kernel, self.bias)

        return outputs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Test orthogonal fully-connected layer
    print("Testing orthogonal fully-connected layer...")

    # batch size
    batch_size = 2

    # exact orthogonal, row-orthogonal, column-orthogonal
    for (in_features, out_features) in [(2048, 10), (10, 2048), (512, 512)]:
        for init in ["uniform", "identical", "reverse", "permutation", "torus"]:
            print("out_features = %d, in_features = %d, init = %s" % (out_features, in_features, init))

            # evaluate the layer with randomized inputs
            inputs  = torch.randn(batch_size, in_features, requires_grad = True).to(device)
            module  = OrthoLinear(in_features = in_features, out_features = out_features, init = init).to(device)
            outputs = module(inputs)

            # check output dimensions
            output_size = outputs.size()
            print("output size: ", output_size)
            assert output_size[0] == batch_size and output_size[1] == out_features

            # check forward norm preservation
            if out_features >= in_features: # column-orthogonal
                norm_inputs  = torch.norm(inputs) 
                norm_outputs = torch.norm(outputs)

                if not torch.isclose(norm_inputs, norm_outputs, rtol = 1e-3, atol = 1e-4):
                    print("norm_inputs: %.4f, norm_outputs: %.4f" % 
                        (norm_inputs.item(), norm_outputs.item()))
                    print("The input norm and output norm do not match.")

            # check backward norm preservation
            if out_features <= in_features: # row-orthogonal
                grad_outputs = torch.randn(batch_size, out_features).to(device)
                grad_inputs  = torch.autograd.grad(outputs, inputs, grad_outputs)[0]
                norm_grad_inputs  = torch.norm(grad_inputs)
                norm_grad_outputs = torch.norm(grad_outputs)

                if not torch.isclose(norm_grad_outputs, norm_grad_inputs):
                    print("norm_grad_inputs: %.4f, norm_grad_outputs: %.4f" % 
                        (norm_grad_inputs.item(), norm_grad_outputs.item()))
                    "The input gradient norm and output gradient norm do not match."

            # check orothogonality by reversion
            if out_features >= in_features: # column-orthogonal
                inputs_ = module(outputs, reverse = True)

                input_size = inputs_.size()
                assert input_size[0] == batch_size and input_size[1] == in_features

                if torch.isclose(torch.norm(inputs - inputs_), torch.tensor(0.).to(device), rtol = 1e-2, atol = 1e-3):
                    print("Success! The restored input matches the original input.")
                else:
                    print("The restored input and the original input do not match.")
                    print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

            print('-' * 60)
