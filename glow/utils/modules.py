import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _single, _pair
from utils.utils import split_feature, compute_same_pad


class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale = 1.0):
        super(_ActNorm, self).__init__()
        
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, inputs):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(inputs.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((inputs.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, inputs, reverse=False):
        if reverse:
            return inputs - self.bias
        else:
            return inputs + self.bias

    def _scale(self, inputs, logdet=None, reverse=False):

        if reverse:
            inputs = inputs * torch.exp(-self.logs)
        else:
            inputs = inputs * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = inputs.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return inputs, logdet

    def forward(self, inputs, logdet=None, reverse=False):
        self._check_inputs_dim(inputs)

        if not self.inited:
            self.initialize_parameters(inputs)

        if reverse:
            inputs, logdet = self._scale(inputs, logdet, reverse)
            inputs = self._center(inputs, reverse)
        else:
            inputs = self._center(inputs, reverse)
            inputs, logdet = self._scale(inputs, logdet, reverse)

        return inputs, logdet


class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale = 1.0):
        super().__init__(num_features, scale)

    def _check_inputs_dim(self, inputs):
        assert len(inputs.size()) == 4
        assert inputs.size(1) == self.num_features, (
            "[ActNorm]: inputs should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, inputs.size()
            )
        )


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = (3, 3),
        stride = (1, 1),
        padding = "same",
        do_actnorm = True,
        weight_std = 0.05,
    ):
        """

        """
        super(Conv2d, self).__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm2d(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, inputs):
        """

        """
        x = self.conv(inputs)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


class LinearZeros(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        logscale_factor = 3
    ):
        """

        """
        super(LinearZeros, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels))

    def forward(self, inputs):
        """

        """
        outputs = self.linear(inputs)
        return outputs * torch.exp(self.logs * self.logscale_factor)


class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = (3, 3),
        stride = (1, 1),
        padding = "same",
        logscale_factor = 3,
    ):
        """

        """
        super(Conv2dZeros, self).__init__()

        if padding == "same":
            padding = compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, inputs):
        """

        """
        outputs = self.conv(inputs)
        return outputs * torch.exp(self.logs * self.logscale_factor)


def gaussian_likelihood(mean, log_std, inputs, reduction = "sum"):
    """
    Compute the log-likelihood of the given data.
    
    Arguments:
    ----------
    mean, log_variance: two (d+2)th-order tensors of size 
        [batch_size, num_features, size_1, ..., size_d]
        Mean and log-variance of the Gaussian distribution.
        Note: log-variance can take any real value.

    inputs: a (d+2)th-order tensor of size
        [batch_size, num_features, size_1, ..., size_d]
        The input data to be evaluated.

    reduction: str (options: "none", "mean", or "sum")
        The reduction mode for different examples.
        Default: "sum"

    Return:
    -------
    lnL: float or a (d+2)th-order tensor of size
        [batch_size, num_features, size_1, ..., size_d]
        The log-likelihood of the input data.
        Note: the lnL is a float if the reduction mode is not "none".

    Notes:
    ------
    var = exp(log_std * 2) --> ln(var) = log_std * 2
    lnL = -1/2 * {ln(2 * PI) + ln(var) + (x - mean)^2 / var}
    
    """
    lnL = -0.5 * (math.log(2 * math.pi) + log_std * 2.0 
        + ((inputs - mean) ** 2) / torch.exp(log_std * 2.0))

    if reduction == "sum":
        lnL = torch.sum(lnL,  dim = list(range(1, lnL.dim())))
    elif reduction == "mean":
        lnL = torch.mean(lnL, dim = list(range(1, lnL.dim())))

    return lnL


def gaussian_sample(mean, log_std, temperature = 1):
    """
    Sample from Gaussian distributions with adjustable temperature.

    Arguments:
    ----------
    mean, log_std: two (d+2)-th order tensors of size
        [batch_size, num_features, size_1, ..., size_d]
        Mean and logartihm of standard deviation of the Gaussian distribution.
        Note: logarithm of standard deviataion can take any real value.

    temperature: float
        The scaling factor for the standard deviation during sampling.
        Note: The temperature is a non-negative floating number.
        Default: 1

    Return:
    -------
    samples: a (d+2)-th order tensor of size
        [batch_size, num_features, size_1, ..., size_d]
        Samples from the Gaussian distributions.

    """
    samples = torch.normal(mean, torch.exp(log_std) * temperature)

    return samples


class Split2d(nn.Module):
    def __init__(self, num_channels):
        """
        Construction of a 2D splitting layer.
        
        Argument:
        ---------
        in_channels: int
            The number of input channels to the splitting layer.

        """
        super().__init__()

        # precessing module before splitting
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def forward(self, inputs, logdet = 0.0, reverse = False, temperature = None):
        """
        Computation of the 2D splitting layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size
            [batch_size, in_channels, height, width]
            The input to the splitting layer.

        input_logdet: float
            Logarithm of the input determinant.
            default: 0.0

        reverse: bool
            Whether to compute the reverse pass of the layer. 
            Default: False

        Returns:
        --------
        outputs: a 4th-order tensor of size
            [batch_size, out_channels, height, width]
            Note: out_channels = in_channels // 2

        output_logdet: float
            Logarithm of the output determinant.

        """
        if not reverse: # normal flow
            z1, z2 = split_feature(inputs, "split")
            mean, logs = split_feature(self.conv(z1), "cross")
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet

        else: # reverse flow
            z1 = inputs
            mean, logs = split_feature(self.conv(z1), "cross")
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim = 1)
            return z, logdet


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle = True):
        """
        Construction of a 2D permutation layer.

        Arguments:
        ----------
        num_channels: int
            The number of input and output channels of the permutation layer.

        shuffle: bool
            Whether to permute the inputs randomly.
            Default: True

        """
        super(Permute2d, self).__init__()
        self.num_channels = num_channels
        self.indices = torch.arange(self.num_channels - 1, -1, -1, dtype = torch.long)
        self.indices_inverse = torch.zeros((self.num_channels), dtype = torch.long)

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        """
        Random generation of the permutation matrix.

        """
        shuffle_idx = torch.randperm(self.indices.shape[0])
        self.indices = self.indices[shuffle_idx]

        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, inputs, logdet = None, reverse = False):
        """
        Computation of the 2D permutation layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size
            [batch_size, num_channels, height, width]
            The inputs to the permutation layer.

        input_logdet: float
            Logarithm of the input determinant.
            default: None

        reverse: bool
            Whether to compute the reverse pass of the layer. 
            Default: False
        
        Returns:
        --------
        outputs: a 4th-order tensor of size
            [batch_size, num_channels, height, width]

        output_logdet: float
            Logarithm of the output determinant.
            default: None

        The layer is determinant invariant, i.e.,
            output_logdet = input_logdet

        """
        assert len(inputs.size()) == 4

        if not reverse:
            inputs = inputs[:, self.indices, :, :]
        else: # if reverse:
            inputs = inputs[:, self.indices_inverse, :, :]

        return inputs, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed = True):
        """
        Construction of the invertible 1x1 convolutional layer.

        Arguments:
        ----------
        num_channels: int
            The number of input and output channels of the invertible layer.

        LU_decomposed: bool
            Whether the weights matrix is stored in its LU decomposed format.
            Default: True

        """
        super(InvertibleConv1x1, self).__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, inputs, reverse = False):
        """
        Construction of the weights for forward or reverse computation.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size 
            [batch_size,  num_channels, height, width]
            The inputs to the invertible layer.

        reverse: bool
            Whether to compute the reverse pass of the layer. 
            Default: False

        Returns:
        --------
        weights: a 4th-order tensor of size
            [input_channels, num_channels, 1, 1]
            The weights for forward or reverse computation.

        """
        b, c, h, w = inputs.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(inputs.device)
            self.eye = self.eye.to(inputs.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, inputs, logdet = None, reverse = False):
        """
        Computation of the invertible 1x1 convolutional layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size 
            [batch_size, num_channels, height, width]
            The inputs to the invertible layer.

        input_logdet: float
            Logarithm of the input determinant.
            default: None

        reverse: bool
            Whether to compute the reverse pass of the layer. 
            Default: False

        Returns:
        --------
        outputs: a 4th-order tensor of size
            [batch_size, num_channels, height, width]
            The outputs of the invertible layer.
        
        output_logdet: float
            Logarithm of the output determinant.
            default: None

        Notes:
        ------
        dlogdet = log|abs(|weights|)| * in_height * in_width.
        If reverse is False:  
            output_logdet = input_logdet + dlogdet
        If reverse is True:
            output_logdet = input_logdet - dlogdet

        """
        weight, dlogdet = self.get_weight(inputs, reverse)

        if not reverse:
            z = F.conv2d(inputs, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(inputs, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class Squeeze2d(nn.Module):
    def __init__(self, factor = 2):
        """
        Construction of a 2D squeezing layer.

        Arguments:
        ----------
        factor: int or tuple(int, int)
            The squeezing factor.
            Note: Given an integer s, it will be repeated into a tuple (s, s)
            Default: 2, or equivalently (2, 2) 

        """
        super(Squeeze2d, self).__init__()
        self.factors = _pair(factor)
        self.factor2 = self.factors[0] * self.factors[1] 

    def forward(self, inputs, logdet = None, reverse = False):
        """
        Compuatation of the 2D squeezing layer.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size
            [batch_size,  input_channels,  input_height,  input_width].
            The inputs to the squeezing layer.

        input_logdet: float
            Logarithm of the input determinant.
            default: None

        reverse: bool
            Whether to compute the reverse pass of the layer. 
            Default: False

        Returns:
        --------
        outputs: a 4th-order tensor of size
            [batch_size, output_channels, output_height, output_width].
            The outputs of the squeezing layer.

        output_logdet: float
            Logarithm of the output determinant.
            default: None

        Notes:
        ------
        If reverse is False:
            output_channels = input_channels // (factors[0] * factors[1])
            output_height = input_height  * factor[0]
            output_width  = input_width   * factor[1]
        If reverse if True:
            output_channels = input_channels  * (factors[0] * factors[1])
            output_height = input_height // factor[0]
            output_width  = input_widht  // factor[1]

        The layer is log-determinant invariant, i.e.,
            output_logdet = input_logdet

        """
        if self.factor2 == 1: return inputs, logdet

        # dimensions of the input tensor
        _, channels, height, width = inputs.size()

        if reverse:
            assert channels % self.factor2 == 0, \
                "The number of channels is not divsible by the product of factors."

            inputs  = inputs.view(-1, channels // self.factor2, 
                self.factors[0], self.factors[1], height, width)
            inputs  = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()

            outputs = inputs.view(-1, channels // self.factor2, 
                height * self.factors[0], width * self.factors[1])

        else: # if not reverse:
            assert height % self.factors[0] == 0 and width % self.factors[1] == 0, \
                "The height or width is not divsible by the corresponding factor."

            inputs  = inputs.view(-1, channels, height // self.factors[0], 
                self.factors[0], width // self.factors[1], self.factors[1])
            inputs  = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()

            outputs = inputs.view(-1, channels * self.factor2, 
                height // self.factors[0], width // self.factors[1])

        return outputs, logdet


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## 2D-Squeeze layer
    print("Testing 2D squeeze layer...")
    print('-' * 60)

    # input dimensions
    batch_size, in_channels, in_height, in_width = 2, 1, 60, 60

    for factors in [(1, 1), (1, 2), (2, 2), (3, 2), (3, 3)]:
        print("factors: (%d, %d)" % (factors[0], factors[1]))

        # compute the expected output dimensions
        out_channels = in_channels * factors[0] * factors[1]
        out_height = in_height // factors[0]
        out_width  = in_width  // factors[1]

        # evaluate the module with randomized inputs
        inputs = torch.randn(batch_size, in_channels, in_height, in_width, requires_grad = True).to(device)
        module = Squeeze2d(factor = factors).to(device)
        outputs, _ = module(inputs, reverse = False)

        # check output dimensions
        out_size = outputs.size()
        print("output size: ", out_size)
        assert  out_size[0] == batch_size and out_size[1] == out_channels \
            and out_size[2] == out_height and out_size[3] == out_width

        # check reversion of the module 
        inputs_, _ = module(outputs, reverse = True)

        in_size = inputs_.size()
        assert  in_size[0] == batch_size and in_size[1] == in_channels \
            and in_size[2] == in_height  and in_size[2] == in_width

        if not torch.isclose(torch.norm(inputs - inputs_), 
            torch.tensor(0.).to(device), rtol = 1e-2, atol = 1e-4):
            print("The restored input and the original input do not match.")
            print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

        print('-' * 60)

    print('')

    ## Invertible 1x1 convolutional layer
    print("Testing invertible 1x1 convolutional layer...")
    print('-' * 60)

    # input dimensions
    batch_size, in_channels, in_height, in_width = 2, 32, 60, 60

    for LU_decomposed in [True, False]:
        print("LU_decomposed = ", LU_decomposed)

        # evaluate the module with randomized inputs
        inputs = torch.randn(batch_size, in_channels, in_height, in_width, requires_grad = True).to(device)
        module = InvertibleConv1x1(num_channels = in_channels, LU_decomposed = LU_decomposed).to(device)
        outputs, _ = module(inputs, reverse = False)

        # check output dimensions
        out_size = outputs.size()
        print("output size: ", out_size)
        assert  out_size[0] == batch_size and out_size[1] == in_channels \
            and out_size[2] ==  in_height and out_size[3] == in_width

        # check reversion of the module
        inputs_, _ = module(outputs, reverse = True)

        in_size = inputs_.size()
        assert  in_size[0] == batch_size and in_size[1] == in_channels \
            and in_size[2] == in_height  and in_size[2] == in_width

        if not torch.isclose(torch.norm(inputs - inputs_),
            torch.tensor(0.).to(device), rtol = 1e-2, atol = 1e-3):
            print("The restored input and the original input do not match.")
            print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

        print('-' * 60)

    print('')

    ## 2D permutation layer
    print("Testing 2D permutation layer...")
    print('-' * 60)

    # input dimensions
    batch_size, in_channels, in_height, in_width = 2, 32, 60, 60

    for shuffle in [True, False]:
        print("shuffle = ", shuffle)

        # evaluate the module with randomized inputs
        inputs = torch.randn(batch_size, in_channels, in_height, in_width, requires_grad = True).to(device)
        module = Permute2d(num_channels = in_channels, shuffle = shuffle).to(device)
        outputs, _ = module(inputs, reverse = False)

        # check output dimensions
        out_size = outputs.size()
        print("output size: ", out_size)
        assert  out_size[0] == batch_size and out_size[1] == in_channels \
            and out_size[2] ==  in_height and out_size[3] == in_width

        # check reversion of the module
        inputs_, _ = module(outputs, reverse = True)

        in_size = inputs_.size()
        assert  in_size[0] == batch_size and in_size[1] == in_channels \
            and in_size[2] == in_height  and in_size[2] == in_width

        if not torch.isclose(torch.norm(inputs - inputs_),
            torch.tensor(0.).to(device), rtol = 1e-2, atol = 1e-4):
            print("The restored input and the original input do not match.")
            print("norm(diff): %.4f" % (torch.norm(inputs - inputs_).item()))

        print('-' * 60)

    print('')

    ## Invertible 2D splitting layer
    print("Testing 2D splitting layer...")
    print('-' * 60)

    # input dimensions
    batch_size, in_channels, in_height, in_width = 2, 32, 60, 60

    # evaluate the module with randomized inputs
    inputs = torch.randn(batch_size, in_channels, in_height, in_width, requires_grad = True).to(device)
    module = Split2d(num_channels = in_channels).to(device)
    outputs, _ = module(inputs, reverse = False)

    out_channels = in_channels // 2

    # check output dimensions
    out_size = outputs.size()
    print("output size: ", out_size)
    assert  out_size[0] == batch_size and out_size[1] == out_channels \
        and out_size[2] ==  in_height and out_size[3] == in_width

    # check reversion of the module
    inputs_, _ = module(outputs, reverse = True, temperature = 1)

    in_size = inputs_.size()
    assert  in_size[0] == batch_size and in_size[1] == in_channels \
        and in_size[2] == in_height  and in_size[2] == in_width

    print('-' * 60)
