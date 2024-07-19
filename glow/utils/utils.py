import math
import torch

def compute_same_pad(kernel_size, stride):
    """
    Compute the padding based on kernel size and stride.

    Arguments:
    ----------
    kernel_size: a tuple of ints of length d
        The kernel size for different dimensions.
        
    stride: a tuple of ints of length d
        The stride for different dimensions.

    Returns:
    --------
    padding: a tuple of ints of length d
        The padding for different dimensions.

    """
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size), \
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]
    

def uniform_binning_correction(x, n_bits = 8):
    """
    Replace x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Arguments:
    ----------
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.

    Returns:
    --------
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).

    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device = x.device)
    return x, objective


def split_feature(tensor, method = "split"):
    """
    Split the features into two parts.

    Arguments:
    ----------
    tensor: a (d+2)th-order tensor of size
        [batch_size, num_channels, ...]
        The input tensor before splitting.

    mothod: str ("split" or "cross")
        The splitting pattern.
        Default: "split"

    Returns:
    --------
    outputs: a tuple of two (d+2)-order tensor of size 
        [batch_size, num_channels // 2, ...]
        The output tensors after splitting.

    """
    num_channels = tensor.size(1)

    if method == "split":
        return tensor[:, : num_channels // 2, ...], tensor[:, num_channels // 2 :, ...]

    elif method == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

    else: # if method not in ["split", "cross"]:
        raise NotImplementedError

