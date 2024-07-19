import math
import torch
import torch.nn as nn

from utils.utils import split_feature, uniform_binning_correction

from utils.modules import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    LinearZeros,
    Squeeze2d,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)

from utils.layers import OrthoConv2d


def get_block(in_channels, out_channels, hidden_channels):
    """
    Construction of a 3-layers bottleneck block.

    Arguments:
    ----------
    in_channels: int
        The number of input channels to the block.
    out_channels: int
        The number of output channels of the block.
    hidden_channels: int
        The number of hidden channels in the block.

    """
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace = False),
        Conv2d(hidden_channels, hidden_channels, kernel_size = (1, 1)),
        nn.ReLU(inplace = False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        actnorm_scale,
        flow_permutation,
        ortho_ker_size,
        ortho_ker_init,
        LU_decomposed,
        flow_coupling,
    ):
        """
        Construction of a block in the Glow model.

        Arguments:
        ----------
        [hyperparameters for input/output channels]
        in_channels: int
            The number of input channels to the block.
        hidden_channels: int
            The number of hidden/output channels of the block.

        [hyperparameters for normalization layers]
        actnorm_scale: float
            The initial scale for the activation normalization.
            Default: 1

        [hyperparameters for flow permutation]
        flow_permutation: str 
            The type of permutation in the Glow model.
            Options: "invconv", "orthoconv", "shuffle", or "reverse"
            Default: "invconv"

        # if "orthoconv" is used
        ortho_ker_size: int
            The kernel size of the orthogonal convolutional layer.
            Note: The argument is applicable if and only if "OrthoConv" is used.
            Default: 3

        ortho_ker_init: int
            The initialization method of the orthogonal convolutional layers.
            Options: "uniform", "identical", "reverse", or "permutation"
            Default: "uniform"
        
        # if "invconv" is used
        LU_decomposed: bool
            Whether the weights in the 1x1 invertible convolutional is LU decomposed.
            Default: True

        [hyperparameters for flow coupling]
        flow_coupling: str
            The type of coupling in the Glow model.
            Options: "additive" or "affine"
            Default: "affine"

        """
        super(FlowStep, self).__init__()

        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

        elif flow_permutation == "orthoconv":
            self.orthoconv = OrthoConv2d(in_channels, in_channels,
                kernel_size = ortho_ker_size, init = ortho_ker_init)
            self.flow_permutation = lambda z, logdet, rev: (self.orthoconv(z, rev), logdet)

        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle = True)
            self.flow_permutation = lambda z, logdet, rev: self.shuffle(z, logdet, rev)

        elif flow_permutation == "reverse":
            self.reverse = Permute2d(in_channels, shuffle = False)
            self.flow_permutation = lambda z, logdet, rev: self.reverse(z, logdet, rev)

        else: # if flow_permutation not in ["invconv", "orthoconv", "shuffle", "reverse"]:
            raise NotImplementedError

        # 3. coupling
        self.flow_coupling = flow_coupling

        if self.flow_coupling == "additive":
            self.block = get_block(in_channels // 2, in_channels // 2, hidden_channels)

        elif self.flow_coupling == "affine":
            self.block = get_block(in_channels // 2, in_channels, hidden_channels)

        else: # if self.flow_coupling not in ["additive", "affine"]:
            raise NotImplementedError

    def forward(self, inputs, logdet = None, reverse = False):
        """
        Computation of a block in the Glow model.

        Arguments:
        ----------
        inputs: a 4th-order tensor of size 
            [batch_size, num_channels, height, width]
            The inputs to the block.

        input_logdet: float
            Logarithm of the input determinant.
            Default: None

        reverse: bool
            Whether to compute the reverse pass of the block.
            Default: False

        Return:
        -------
        outputs: a 4th-order tensor of size 
            [batch_size, num_channels, height, width]
            The outputs of the block.

        output_logdet: float        
            Logarithm of the outptu determinant.
            Default: None

        """
        if not reverse: # normal flow
            return  self.normal_flow(inputs, logdet)
        else: # reverse flow
            return self.reverse_flow(inputs, logdet)

    def normal_flow(self, inputs, logdet):
        """
        Forward pass of the block.

        """
        assert inputs.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(inputs, logdet = logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")

        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)

        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet

        else: # if self.flow_coupling not in ["additive", "affine"]:
            raise NotImplementedError

        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, inputs, logdet):
        """
        Reverse pass of the block.

        """
        assert inputs.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(inputs, "split")

        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)

        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim = [1, 2, 3]) + logdet

        else: # if self.flow_coupling not in ["additive", "affine"]:
            raise NotImplementedError

        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet = logdet, reverse = True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        num_scales,
        num_blocks,
        actnorm_scale,
        flow_permutation,
        ortho_ker_size,
        ortho_ker_init,
        LU_decomposed,
        flow_coupling
    ):
        """
        Construction of the Glow network.

        Arguments:
        ----------
        [hyperparameters for network architecture]
        image_shape: a tuple of (int, int, int)
            The height, width, and channels of the input images.
        hidden_channels: int
            The number of hidden/output channels of each block.
        num_scales: int
            The number of scales in the multi-scale architecture.
            Default: 3
        num_blocks: int
            The number of blocks for each scale.
            Default: 32

        [hyperparameters for normalization layers]
        actnorm_scale: float
            The initial scale for the activation normalization.
            Default: 1

        [hyperparameters for flow permutation]
        flow_permutation: str 
            The type of permutation in the Glow model.
            Options: "invconv", "orthoconv", "shuffle", or "reverse"
            Default: "invconv"

        # if "orthoconv" is used
        ortho_ker_size: int
            The kernel size of the orthogonal convolutional layers.
            Note: Only odd kernel size is supported.
            Default: 3

        ortho_ker_init: int
            The initialization method of the orthogonal convolutional layers.
            Options: "uniform", "identical", "reverse", or "permutation"
            Default: "uniform"
        
        # if "invconv" is used
        LU_decomposed: bool
            Whether the weights in the 1x1 invertible convolutional is LU decomposed.
            Default: True
        
        [hyperparameters for flow coupling]
        flow_coupling: str 
            The type of coupling in the Glow model.
            Options: "additive" or "affine"
            Default: "affine"
            
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        H, W, C = image_shape

        for i in range(num_scales):

            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(Squeeze2d(factor = 2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(num_blocks):
                self.layers.append(
                    FlowStep(
                        in_channels = C,
                        hidden_channels = hidden_channels,
                        actnorm_scale = actnorm_scale,
                        flow_permutation = flow_permutation,
                        ortho_ker_size = ortho_ker_size,
                        ortho_ker_init = ortho_ker_init,
                        LU_decomposed = LU_decomposed,
                        flow_coupling = flow_coupling
                    )
                )
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < num_scales - 1:
                self.layers.append(Split2d(num_channels = C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, inputs, logdet = 0.0, reverse = False, temperature = None):
        """
        Computation of the Glow network.
    
        Arguments:
        ----------
        inputs: a 4th-order tensor of size
            [batch_size, img_channels, img_height, img_width]
            The input images to the Glow network.

        input_logdet: float
            Logarithm of the input determinant.
            Default: 0.0
    
        reverse: bool
            Whether to compute the reverse pass of the block.
            Default: False

        temperature: float
            The scaling factor for the standard deviation during sampling.
            Note: The temperature is a non-negative floating number.
            Default: None

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, out_height, out_width]
            The output 

        output_logdet: float        
            Logarithm of the outptu determinant.
            Default: None

        """
        if not reverse: # normal flow 
            return self.encode(inputs, logdet)
        else: # reverse flow
            return self.decode(inputs, temperature)

    def encode(self, z, logdet = 0.0):
        """
        Forward pass of the Glow network.

        """
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse = False)
            
        return z, logdet

    def decode(self, z, temperature = None):
        """
        Reverse pass of the Glow network.

        """
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet = 0, reverse = True, temperature = temperature)
            else:
                z, logdet = layer(z, logdet = 0, reverse = True)

        return z


class Glow(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        num_scales,
        num_blocks,
        actnorm_scale,
        flow_permutation,
        ortho_ker_size,
        ortho_ker_init,
        LU_decomposed,
        flow_coupling,
        y_classes,
        learn_top,
        y_condition
    ):
        """

        """
        super().__init__()

        self.flow = FlowNet(
            image_shape = image_shape,
            hidden_channels = hidden_channels,
            num_scales = num_scales,
            num_blocks = num_blocks,
            actnorm_scale = actnorm_scale,
            flow_permutation = flow_permutation,
            ortho_ker_size = ortho_ker_size,
            ortho_ker_init = ortho_ker_init,
            LU_decomposed = LU_decomposed,
            flow_coupling = flow_coupling
        )

        self.y_classes = y_classes
        self.y_condition = y_condition

        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(y_classes, 2 * C)
            self.project_class = LinearZeros(C, y_classes)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )

    def prior(self, data, y_onehot = None):
        """

        """
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(32, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)

        return split_feature(h, "split")

    def forward(self, x = None, y_onehot = None, z = None, temperature = None, reverse = False):
        """
        

        """
        if not reverse: # normal flow
            return  self.normal_flow(x, y_onehot)
        else: # reverse flow
            return self.reverse_flow(z, y_onehot, temperature)

    def normal_flow(self, x, y_onehot):
        """

        """
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse = False)

        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)

        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature):
        """

        """
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature = temperature, reverse = True)
        return x

    def set_actnorm_init(self):
        """

        """
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
