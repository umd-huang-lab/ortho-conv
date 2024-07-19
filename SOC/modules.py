import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import einops
import numpy as np

try:
    import sys
    sys.path.append('lconvnet')
    from lconvnet.layers import RKO  as _RKO
except:
    pass

from lconvnet.layers import (
    BjorckLinear, DilatedBjorckConv2d as _BjorckConv,
    CayleyLinear, CayleyConv as _CayleyConv,
    LieExpLinear, LieExpConv as _LieExpConv,
    Paraunitary as _SCFac, Paraunitary,
    BCOP as _BCOP,
    SVCM as _SVCM,
    OSSN as _OSSN,
    RKO  as _RKO,
)


# Extend this class to get emulated striding (for stride 2 only)
class StridedConv(nn.Module):
    def __init__(self, *args, **kwargs):
        striding = False
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            kwargs['stride'] = 1
            striding = True
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2 # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
            args = tuple(args)
        else: # handles OSSN case
            if len(args) == 3:
                kwargs['padding'] = args[2] // 2
            else:
                kwargs['padding'] = kwargs['kernel_size'] // 2
        super().__init__(*args, **kwargs)
        downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
        if striding:
            self.register_forward_pre_hook(lambda _, x: \
                    einops.rearrange(x[0], downsample, k1=2, k2=2))


class SCFac(StridedConv, _SCFac):
    pass

class SVCM(StridedConv, _SVCM):
    pass

class OSSN(StridedConv, _OSSN):
    pass

class RKO(StridedConv, _RKO):
    pass

class BCOP(StridedConv, _BCOP):
    pass

class CayleyConv(StridedConv, _CayleyConv):
    pass

class LieExpConv(StridedConv, _LieExpConv):
    pass

class BjorckConv(StridedConv, _BjorckConv):
    pass
