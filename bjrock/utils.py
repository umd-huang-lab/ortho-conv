import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


def _circular_pad(inputs: Tensor, padding: int) -> Tensor:
    """
    Computation of circular padding of convolution kernels.

    Arguments:
    ----------
    inputs: a (d+2)-order (d <= 3) tensor of size
        [batch_size, num_channels, input_size_1, ..., input_size_d] 
        The input before circular padding.

    padding: a tuple of ints with length 2*d (d <= 3)
        (padding_front_1, padding_rear_1, ..., padding_front_d, padding_rear_d)
        The number of entries added to each side of the inputs.

    Return:
    -------
    outputs: a (d+2)-order (d <= 3) tensor of size
        [batch_size, num_channels, output_size_1, ..., output_size_d]
        The output after circular padding.
        Note: output_size_i = input_size_i + padding_front_i + padding_rear_i

    """
    if padding[-1] > 0:
        inputs = torch.cat([inputs, inputs[:, :, 0:padding[-1]]], dim = 2)
        inputs = torch.cat([inputs[:, :, -(padding[-1] + padding[-2]):-padding[-1]], inputs], dim = 2)
    elif padding[-2] > 0:
        inputs = torch.cat([inputs[:, :, -padding[-2]:], inputs], dim = 2)

    if len(padding) > 2:
        if padding[-3] > 0:
            inputs = torch.cat([inputs, inputs[:, :, :, 0:padding[-3]]], dim = 3)
            inputs = torch.cat([inputs[:, :, :, -(padding[-3] + padding[-4]):-padding[-3]], inputs], dim = 3)
        elif padding[-4] > 0:
            inputs = torch.cat([inputs[:, :, :, -padding[-4]:], inputs], dim = 3)

    if len(padding) > 4:
        if padding[-5] > 0:
            inputs = torch.cat([inputs, inputs[:, :, :, :, 0:padding[-5]]], dim = 4)
            inputs = torch.cat([inputs[:, :, :, :, -(padding[-5] + padding[-6]):-padding[-5]], inputs], dim = 4)
        elif padding[-5] > 0:
            inputs = torch.cat([inputs[:, :, :, :, -padding[-6]:], inputs], dim = 4)

    return inputs


def _power_iteration(
    matrix: Tensor,
    iters: int = 10,
    thres: float = 1e-6,
    init_v: Tensor = None,
    return_v: bool = True,
) -> Tensor:
    """
    Power iteration of a matrix (fully-connected layer).

    Arguments:
    ----------
    matrix: a matrix of size [out_features, in_features]
        The unnormalized matrix to power iteration.
    
    [hyper-parameters for power iteration]
    iters: int
        The number of steps for power iteration.
        Default: 10
    thres: float
        The absolute tolerance of approximation.
        Default: 1e-6

    init_v: a vector of length [in_features]
        The initialization of the right singular vector.
        Default: None
    return_v: bool
        Whether to return the top right singular vector.
        Default: True

    Returns:
    --------
    scale: float
        The top singular value of the matrix.

    v_vec: a vector of length of [in_features]
        The top right singular vector of the matrix. 

    """
    [out_features, in_features], device = matrix.shape, matrix.device

    # initialization of the right singular vector
    if init_v is None:
        init_v = torch.randn(in_features, device = device)
    else: # if init_u is not None
        assert tuple(init_v.shape) == (in_features, )

    init_v = F.normalize(init_v, dim = 0)

    # power iteration
    for _ in range(iters):
        u_vec = F.normalize(torch.mv(matrix, init_v), dim = 0)
        v_vec = F.normalize(torch.mv(matrix.t(), u_vec), dim = 0)

        if torch.linalg.norm(v_vec - init_v) < thres: break
        init_v = v_vec

    # top singular value
    scale = torch.dot(u_vec, torch.mv(matrix, v_vec))

    if return_v:
        return scale, v_vec
    else: # if return_uv:
        return scale


def _bjorck_orthonormalize(
    matrix: Tensor,
    order: int = 1,
    iters: int = 20,
    thres: float = 1e-6
) -> Tensor:
    """
    Björck orthonormalization of a matrix (fully-connected layer).

    Arguments:
    ----------
    matrix: a matrix of size [out_features, in_features]
        The input normalized matrix to Björck orthonormalization.
    
    [hyper-parameters for Björck orthonormalization]
    order: int
        The order of Taylor's expansion.
        Default: 1
    iters: int
        The maximum number of iterations.
        Default: 20
    thres: float
        The absolute tolerance of approximation.
        Default: 1e-6

    Return:
    -------
    matrix: a matrix of size [out_features, in_features]
        The output orthogonal matrix of Björck orthonormalization.

    """
    [out_features, in_features] = matrix.shape

    # return row-orthogonal matrix
    if out_features < in_features:
        return _bjorck_orthonormalize(
            matrix.transpose(0, 1),
            order = order, iters = iters, thres = thres
        ).transpose(0, 1)

    # identical mapping for matrix multiplication
    identity = torch.eye(in_features, device = matrix.device)

    # Björck orthonormalization
    for _ in range(iters):
        factor = identity - torch.matmul(matrix.t(), matrix)
        taylor = identity + 0.5 * factor

        if order > 1:
            residual = factor
            p, q, sgn = 1.0, -0.5, -1.0
            for i in range(2, order + 1):
                p, q, sgn = p * (i + 1.0), q * (-i - 0.5), -sgn
                residual = torch.matmul(residual, factor)
                taylor = taylor + sgn * (q / p) * residual
                
        matrix = torch.matmul(matrix, taylor)

        if torch.linalg.norm(factor) < thres: break

    return matrix


def _power_iteration_conv1d_circ(
    kernel: Tensor,
    iters: int = 10,
    thres: float = 1e-6,
    init_v: Tensor = None,
    return_v: bool = True,
) -> Tensor:
    """
    Power iteration of a 1D-convolution layer.

    Arguments:
    ----------
    kernel: a 3rd-order tensor of size
        [out_channels, in_channels, kernel_length].
        The unnormalized convolution kernel to power iteration.

    [hyper-parameters for power iteration]
    iters: int
        The number of steps for the power iteration.
        Default: 10
    thres: float
        The absolute tolerance of the approximation.
        Default: 1e-6

    init_v: a 3rd-order tensor of size
        [1, in_channels, kernel_length]
        The initialization of the right singular vector.
        Default: None
    return_v: bool
        Whether to return the top right singular vector.
        Default: False

    Return:
    -------
    scale: scalar
        The (approximated) operator norm of the convolution kernel.

    v_vec: a 3rd-order tensor of size
        [1, in_channels, kernel_length]
        The top right singular vector of the convolution kernel.

    """
    [out_channels, in_channels, length], device = kernel.shape, kernel.device

    # functionals for 1D-padding
    v_pad = lambda inputs: _circular_pad(inputs, (length // 2, (length - 1) // 2))
    u_pad = lambda inputs: _circular_pad(inputs, ((length - 1) // 2, length // 2))

    # initialization of the right singular vector
    if init_v is None:
        init_v = torch.randn(1, in_channels, length, device = device)
    else: # if init_v is not None:
        assert tuple(init_v.shape) == (1, in_channels, length)

    # power iteration for 1D-convolution
    init_v = init_v / torch.linalg.norm(init_v)
    for _ in range(iters):
        u_vec = F.conv1d(v_pad(init_v), kernel)
        u_vec = u_vec / torch.linalg.norm(u_vec)
        v_vec = F.conv_transpose1d(u_pad(u_vec), kernel, padding = length - 1)
        v_vec = v_vec / torch.linalg.norm(v_vec)

        if torch.linalg.norm(v_vec - init_v) < thres: break
        init_v = v_vec

    # top singular value
    scale = torch.dot(u_vec.view(-1), F.conv1d(v_pad(v_vec), kernel).view(-1))

    if return_v:
        return scale, v_vec
    else: # if not return_v:
        return scale


def _bjorck_orthonormalize_conv1d_circ(
    kernel: Tensor,
    order: int = 1,
    iters: int = 20,
    thres: float = 1e-6
) -> Tensor:
    """
    Bjorck orthonormalization of a dilated 1D-convolution layer.

    Arguments:
    ----------
    kernel: a 3rd-order tensor of size
        [outhannels, in_channels, length].
        The input normalized kernel to Björck orthonormalization.

    [hyper-parameters for Björck orthonormalization]
    order: int
        The order of Taylor's expansion.
        Default: 1
    iters: int
        The maximum number of iterations.
        Default: 20
    thres: float
        The absolute tolerance of the approximation.
        Default: 1e-6

    Return:
    -------
    kernel: a 3rd-order tensor of size
        [out_channels, in_channels, length].
        The output orthogonal kernel of Björck orthonormalization.

    """
    [out_channels, in_channels, length], device = kernel.shape, kernel.device

    # return row-orthogonal kernel
    if out_channels < in_channels:
        return _bjorck_orthonormalize_conv1d_circ(
            kernel.transpose(0, 1),
            iters = iters, thres = thres,
        ).transpose(0, 1)

    # functionals for convolution (circular padding)
    _conv1d = lambda filters, inputs: F.conv1d(_circular_pad(
        inputs, (length // 2, (length - 1) // 2)), filters)
    _corr1d = lambda filters, inputs: F.conv_transpose1d(_circular_pad(
        inputs, ((length - 1) // 2, length // 2)), filters, padding = length - 1)

    # identity for convolution
    identity = torch.zeros(in_channels, in_channels, length, device = device)
    identity[:, :, length // 2] = torch.eye(in_channels, device = device)

    # Björck orthonormalization
    kernel = kernel.transpose(0, 1)

    for k in range(iters):
        factor = identity - _conv1d(kernel, kernel)
        taylor = identity + 0.5 * factor

        if order > 1:
            residual = factor
            p, q, sgn = 1.0, -0.5, -1.0
            for i in range(2, order + 1):
                p, q, sgn = p * (i + 1.0), q * (-i - 0.5), -sgn
                residual = _corr1d(residual, factor)
                taylor = taylor + sgn * (q / p) * residual

        kernel = _corr1d(kernel, taylor)

        if torch.linalg.norm(factor) < thres: break

    return kernel.transpose(0, 1)


def _power_iteration_conv2d_circ(
    kernel: Tensor,
    iters: int = 10,
    thres: float = 1e-6,
    init_v: Tensor = None,
    return_v: bool = True,
) -> Tensor:
    """
    Power iteration of a 2D-convolution layer.

    Arguments:
    ----------
    kernel: a 4th-order tensor of size
        [out_channels, in_channels, kernel_height, kernel_width].
        The unnormalized kernel to power iteration.
    
    [hyper-parameters for power iteration]
    iters: int
        The number of steps for power iteration.
        Default: 10
    thres: float
        The absolute tolerance of approximation.

    init_v: int
        The initialization of the right singular vector.
        Default: None
    return_v: bool
        Whether to return the top right singular vector.
        Default: True

    Return:
    -------
    scale: scalar
        The (approximated) operator norm of the convolution kernel.

    v_vec: a 4-th order tensor of size
        [1, in_channels, kernel_height, kernel_width]
        The top right singular vector of the convolution kernel.

    """
    [out_channels, in_channels, height, width] = kernel.shape

    # functionals for 2D-padding
    v_pad = lambda inputs: _circular_pad(inputs, 
        (width // 2, (width - 1) // 2, height // 2, (height - 1) // 2))
    u_pad = lambda inputs: _circular_pad(inputs,
        ((width - 1) // 2, width // 2, (height - 1) // 2, height // 2))

    # initialization for the right singular vector
    if init_v is None:
        v_vec = torch.randn(1, in_channels, height, width, device = kernel.device)
    else: # if init_v is None:
        assert tuple(init_v.shape) == (1, in_channels, height, width)

    # power iteration for 2D-convolution
    init_v = init_v / torch.linalg.norm(init_v)
    for _ in range(iters):
        u_vec = F.conv2d(v_pad(init_v), kernel)
        u_vec = u_vec / torch.linalg.norm(u_vec)
        v_vec = F.conv_transpose2d(u_pad(u_vec), kernel, padding = (height - 1, width - 1))
        v_vec = v_vec / torch.linalg.norm(v_vec)

        if torch.linalg.norm(v_vec - init_v) < thres: break
        init_v = v_vec

    # top singular value
    scale = torch.dot(u_vec.view(-1), F.conv2d(v_pad(v_vec), kernel).view(-1))

    if return_v:
        return scale, v_vec
    else: # if not return_uv:
        return scale


def _bjorck_orthonormalize_conv2d_circ(
    kernel: Tensor,
    order: int = 1,
    iters: int = 10,
    thres: float = 1e-6
) -> Tensor:
    """
    Bjorck orthonormalization of a dilated 2D-convolution layer.

    Arguments:
    ----------
    kernel: a 4th-order tensor of size
        [out_channels, in_channels, kernel_height, kernel_width]
        The input normalized kernel to Björck orthonormalization.

    [hyper-parameters for Björck orthonormalization]
    order: int
        The order of Taylor's expansion.
        Default: 1
    iters: int
        The maximum number of iterations.
        Default: 20
    thres: float
        The absolute tolerance of approximation.
        Default: 1e-6

    Return:
    -------
    kernel: a 4th-order tensor of size
        [out_channels, in_channels, kernel_height, kernel_width].
        The output orthogonal kernel of Björck orthonormalization.

    """
    [out_channels, in_channels, height, width], device = kernel.shape, kernel.device

    # row-orthogonal kernel
    if out_channels < in_channels:
        return _bjorck_orthonormalize_conv2d_circ(
            kernel.transpose(0, 1),
            iters = iters, thres = thres
        ).transpose(0, 1)

    # functionals for convolution (circular padding)
    _conv2d = lambda filters, inputs: F.conv2d(_circular_pad(
        inputs, (width // 2, (width - 1) // 2, height // 2, (height - 1) // 2)), filters)
    _corr2d = lambda filters, inputs: F.conv_transpose2d(_circular_pad(
        inputs, ((width - 1) // 2, width // 2, (height - 1) // 2, height // 2)), filters,
        padding = (height - 1, width - 1))

    # identity for convolution
    identity = torch.zeros(in_channels, in_channels, height, width, device = device)
    identity[:, :, height // 2, width // 2] = torch.eye(in_channels, device = device)

    # Björck orthonormalization
    kernel = kernel.transpose(0, 1)

    for k in range(iters):
        factor = identity - _conv2d(kernel, kernel)
        taylor = identity + 0.5 * factor

        if order > 1:
            residual = factor
            p, q, sgn = 1.0, -0.5, -1.0
            for i in range(2, order + 1):
                p, q, sgn = p * (i + 1.0), q * (-i - 0.5), -sgn
                residual = _corr2d(residual, factor)
                taylor = taylor + sgn * (q / p) * residual

        kernel = _corr2d(kernel, taylor)

        if torch.linalg.norm(factor) < thres: break

    return kernel.transpose(0, 1)


# def _bjorck_orthonormalize_conv1d(
#     kernel: Tensor,
#     iters: int = 500,
#     thres: float = 1e-6
# ) -> Tensor:
#     """
#     Bjorck orthonormalization of a 1D-convolution layer.

#     Arguments:
#     ----------
#     kernel: a 3rd-order tensor of size
#         [outhannels, in_channels, length].
#         The input normalized kernel to Björck orthonormalization.

#     [hyper-parameters for Björck orthonormalization]
#     iters: int
#         The maximum number of iterations.
#         Default: 100
#     thres: float
#         The absolute tolerance of approximation.
#         Default: 1e-6

#     Return:
#     -------
#     kernel: a 3rd-order tensor of size
#         [out_channels, in_channels, length].
#         The output orthogonal kernel of Björck orthonormalization.

#     """
#     [out_channels, in_channels, length], device = kernel.shape, kernel.device

#     # return row-orthogonal kernel
#     if out_channels < in_channels:
#         return _bjorck_orthonormalize_conv1d(
#             kernel.transpose(0, 1),
#             iters = iters, thres = thres,
#         ).transpose(0, 1)

#     # functionals for convolution (circular padding)
#     _conv1d = lambda filters, inputs: F.conv1d(F.pad(
#         inputs, (length // 2, (length - 1) // 2)), filters)
#     _conv_transpose1d = lambda filters, inputs: F.conv_transpose1d(F.pad(
#         inputs, ((length - 1) // 2, length // 2)), filters, padding = length - 1)

#     # functional for circular convoluiton
#     _conv1d_ = lambda filters, inputs: F.conv1d(_circular_pad(
#         inputs, (length // 2, (length - 1) // 2)), filters)
#     _conv_transpose1d_ = lambda filters, inputs: F.conv_transpose1d(_circular_pad(
#         inputs, ((length - 1) // 2, length // 2)), filters, padding = length - 1)

#     # identity for convolution
#     identity = torch.zeros(in_channels, in_channels, length, device = device)
#     identity[:, :, length // 2] = torch.eye(in_channels, device = device)

#     # Björck orthonormalization
#     kernel = kernel.transpose(0, 1)

#     for k in range(iters):
#         # circular convolution
#         factor = identity - _conv1d_(kernel, kernel)
#         kernel += 0.5 * _conv_transpose1d_(kernel, factor)

#         # standard convolution
#         factor = identity - _conv1d(kernel, kernel)
#         kernel += 0.5 * _conv_transpose1d(kernel, factor)

#         # termination condition
#         if torch.norm(factor) < 1e-6: 
#             break

#     return kernel.transpose(0, 1)


# def _bjorck_orthonormalize_conv2d(
#     kernel: Tensor,
#     iters: int = 500,
#     thres: float = 1e-5
# ) -> Tensor:
#     """
#     Bjorck orthonormalization of a 2D-convolution layer.

#     Arguments:
#     ----------
#     kernel: a 4th-order tensor of size
#         [out_channels, in_channels, kernel_height, kernel_width]
#         The input normalized kernel to Björck orthonormalization.

#     [hyper-parameters for Björck orthonormalization]
#     iters: int
#         The maximum number of iterations.
#         Default: 100
#     thres: float
#         The absolute tolerance of the approximation.
#         Default: 1e-5

#     Return:
#     -------
#     kernel: a 4th-order tensor of size
#         [out_channels, in_channels, kernel_height, kernel_width].
#         The output orthogonal kernel of Björck orthonormalization.

#     """
#     [out_channels, in_channels, height, width] = kernel.shape

#     # row-orthogonal kernel
#     if out_channels < in_channels:
#         return _bjorck_orthonormalize_conv2d(
#             kernel.transpose(0, 1),
#             iters = iters, thres = thres
#         ).transpose(0, 1)

#     # functionals for convolution (zero padding)
#     _conv2d_zero = lambda filters, inputs: F.conv2d(F.pad(
#         inputs, (height // 2, (height - 1) // 2, width // 2, (width - 1) // 2)), filters)
#     _corr2d_zero = lambda filters, inputs: F.conv_transpose2d(F.pad(
#         inputs, ((height - 1) // 2, height // 2, (width - 1) // 2, width // 2)), filters,
#         padding = (height - 1, width - 1))

#     # functionals for convolution (circular padding)
#     _conv2d_circ = lambda filters, inputs: F.conv2d(_circular_pad(
#         inputs, (height // 2, (height - 1) // 2, width // 2, (width - 1) // 2)), filters)
#     _corr2d_circ = lambda filters, inputs: F.conv_transpose2d(_circular_pad(
#         inputs, ((height - 1) // 2, height // 2, (width - 1) // 2, width // 2)), filters,
#         padding = (height - 1, width - 1))

#     # functionals for standard convolution
#     _conv2d = lambda filters, inputs: F.conv2d(
#         inputs, filters, padding = (height - 1, width - 1))
#     _corr2d = lambda filters, inputs: F.conv_transpose2d(
#         inputs, filters, padding = (height - 1, width - 1))

#     # identity for convolution
#     identity_ = torch.zeros(in_channels, in_channels, height, width, device = kernel.device)
#     identity_[:, :, height // 2, width // 2] = torch.eye(in_channels, device = kernel.device)

#     identity = torch.zeros(in_channels, in_channels, 2 * height - 1, 2 * width - 1, device = kernel.device)
#     identity[:, :, height - 1, width - 1] = torch.eye(in_channels, device = kernel.device)

#     # Björck orthonormalization
#     kernel = kernel.transpose(0, 1)

#     for k in range(iters):
#         # # convolution (circular padding)
#         # Q_circ = identity_ - _conv2d_circ(kernel, kernel)
#         # kernel += 0.5 * _corr2d_circ(kernel, Q_circ)

#         # # # convolution (zero padding)
#         # Q_zero = identity_ - _conv2d_zero(kernel, kernel)
#         # kernel += 0.5 * _corr2d_zero(kernel, Q_zero)

#         # standard convolution
#         factor = identity - _conv2d(kernel, kernel)
#         kernel += 0.5 * _corr2d(kernel, factor)

#         # termination condition
#         if torch.norm(factor) < 1e-5: break

#     return kernel.transpose(0, 1)
