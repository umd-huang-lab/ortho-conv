import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _bjorck_orthonormalize

from torch import Tensor
from typing import Callable


class BjorckLinear(nn.Module):

    def __init__(
        self,
        in_features:  int, 
        out_features: int,
        bias: bool = False,
        init: str = "permutation",
        order: int = 1,
        iters: int = 10,
        thres: float = 1e-6
    ) -> None:
        """
        Orthogonal linear layer with Björck orthonormalization.
        
        Arguments:
        ----------
        [hyper-parameters for linear layer]
        in_features: int
            The number of input features to the layer.
        out_features: int
            The number of output features of the layer.
        bias: bool
            Whether to add a learnable bias to the output.
            Default: False

        init: str ("identical", "reverse", or permutation")
            The initialization method of the orthogonal layer.
            Default: "permutation"

        [hyper-parameters for Björck orthonormalization]
        order: int
            The order of Taylor's expansion for Björck orthonormalization.
            Default: 1
        iters: int
            The maximum iterations for Björck orthonormalization.
            Default: 100
        thres: float
            The absolute tolerance for Björck orthonormalization.
            Default: 1e-6

        """
        super(BjorckLinear, self).__init__()

        ## [hyper-parameters for the Björck linear layer]
        self.in_features  =  in_features
        self.out_features = out_features

        if self.in_features > self.out_features:
            warnings.warn("The layer is made row-orthogonal.")
        elif self.in_features < self.out_features: 
            warnings.warn("The layer is made column-orthogonal.")

        self.order = order
        self.iters = iters
        self.thres = thres

        ## [initialization of the layer parameters]
        self.weight = nn.Parameter(torch.zeros(self.out_features, self.in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else: # if not bias:
            self.register_parameter("bias", None)

        self._initialize(init)

        ## [functionals for forward/inverse computation]
        self._forward = lambda inputs, weights, biases: F.linear(inputs, weights, biases)

        if bias:
            self._inverse = lambda inputs, weights, biases: F.linear(inputs - biases, weights.t())
        else: # if not bias:
            self._inverse = lambda inputs, weights, biases: F.linear(inputs, weights.t())

    def _initialize(self, init: str = "permutation") -> None:
        """
        Initialization of the Björck linear layer.

        Argument:
        ---------
        init: str ("identical", "reverse", or "permutation")
            The initialization method of the Björck linear layer.
            Default: "permutation"

        """
        if init not in ["identical", "reverse", "permutation"]:
            raise ValueError("The initialization method %s is not supported." % init)

        max_features = max(self.out_features, self.in_features)
        min_features = min(self.out_features, self.in_features)

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

        self.weight.data = matrix

    def _project(self) -> None:
        """
        Björck orthonormalization of the weights.
        
        """
        with torch.no_grad():
             self.weight.data = _bjorck_orthonormalize(
                self.weight.data, self.order, self.iters, self.thres)

    def forward(self, inputs: Tensor, projection: bool = True, inverse: bool = False) -> Tensor:
        """
        Computation of the BjorckLinear layer. 
        
        Arguments:
        ----------
        inputs: a matrix of size [batch_size, in_features]
            The input to the Björck linear layer.

        projection: bool
            Whether to orthonormalize the weights before computation.
            Default: True

        inverse: bool
            Whether to compute the inverse of the layer.
            Default: False

        Return:
        -------
        outputs: a matrix of size [batch_size, out_features]
            The output of the Björck linear layer.

        """
        if self.training and projection: 
            self._project()

        if not inverse:
            outputs = self._forward(inputs, self.weight, self.bias)
        else: # if inverse:
            outputs = self._inverse(inputs, self.weight, self.bias)

        return outputs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Testing orthogonal linear layer
    print("Testing BjorckLinear layer.")
    print('*' * 60)

    # hyper-parameters
    batch_size = 1
    order, iters, thres = 1, 10, 1e-6

    # exact orthogonal, row-orthogonal, column-orthogonal
    for (in_features, out_features) in [(24, 12), (12, 24), (16, 16)]:
        for init in ["identical", "reverse", "permutation"]:
            print("out_features = %d, in_features = %d" % (out_features, in_features))
            print("initialization = %s" % init)

            # evaluate the layer with randomized inputs
            inputs  = torch.randn(batch_size, in_features, requires_grad = True).to(device)
            module  = BjorckLinear(in_features = in_features, out_features = out_features, 
                bias = False, init = init, order = order, iters = iters, thres = thres).to(device)
            outputs = module(inputs, projection = True, inverse = False)

            # check output dimensions
            output_size = outputs.size()
            print("output size: ", output_size)
            assert output_size == (batch_size, out_features)

            # 1) check forward norm preservation
            if out_features >= in_features: # column-orthogonal
                norm_inputs  = torch.norm(inputs) 
                norm_outputs = torch.norm(outputs)

                if torch.isclose(norm_inputs, norm_outputs, rtol = 1e-3, atol = 1e-4):
                    print("Success! The output norm matches the input norm.")
                else:
                    print("norm_inputs: %.4f, norm_outputs: %.4f" % 
                        (norm_inputs.item(), norm_outputs.item()))
                    print("The output and input norms do not match.")

            # 2) check backward norm preservation
            if out_features <= in_features: # row-orthogonal
                grad_outputs = torch.randn(batch_size, out_features).to(device)
                grad_inputs  = torch.autograd.grad(outputs, inputs, grad_outputs)[0]
                norm_grad_inputs  = torch.norm(grad_inputs)
                norm_grad_outputs = torch.norm(grad_outputs)

                if torch.isclose(norm_grad_outputs, norm_grad_inputs, rtol = 1e-3, atol = 1e-4):
                    print("Success! The input gradient norm matches the output gradient norm.")
                else:
                    print("norm_grad_inputs: %.4f, norm_grad_outputs: %.4f" %
                        (norm_grad_inputs.item(), norm_grad_outputs.item()))
                    print("The input and output gradient norms do not match.")

            # 3) check orothogonality by inversion
            if out_features >= in_features: # column-orthogonal
                module.eval()
                inputs_ = module(outputs, projection = False, inverse = True)

                input_size = inputs_.size()
                assert input_size == (batch_size, in_features)

                if torch.isclose(torch.norm(inputs - inputs_), torch.tensor(0.).to(device), rtol = 1e-2, atol = 1e-3):
                    print("Success! The restored input matches the original input.")
                else:
                    print("The restored input and original input do not match.")
                    print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

            print('-' * 60)
