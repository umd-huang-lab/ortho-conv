import torch
import torch.nn as nn


class Matrix(nn.Module):

    def __init__(self, num_cols, num_rows):
        """
        Initialization of a matrix container.

        Arguments:
        ----------
        num_cols: int
            The number of columns in the matrix.
        num_rows: int
            The number of rows in the matrix.

        """
        super(Matrix, self).__init__()

        self.matrix = nn.Parameter(torch.zeros(num_rows, num_cols))

    def forward(self, inputs, adjoint = True):
        """
        Computation of a matrix multiplication.

        Argument:
        ---------
        inputs: a matrix of size [num_cols, rank] (or [num_rows, rank])
            The inputs to the matrix multiplication.

        right: bool
            Whether to multiply with the adjoint matrix.
            default: False

        Return:
        -------
        outputs: a matrix of length [num_rows, rank] (or [num_cols, rank]
            The outputs of the matrix multiplication.

        """
        if not adjoint:
            outputs = torch.matmul(self.matrix, inputs)
        else: # if adjoint:
            outputs = torch.matmul(torch.transpose(self.matrix, 0, 1), inputs)

        return outputs
