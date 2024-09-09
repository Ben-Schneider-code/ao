import torch
from torch.nn.common_types import _size_2_t
from typing import Union
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import functools
from torch.nn import init

class SCConv2d(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        b: float = 2.0, # bit depth
        e: float = -8.0, # scaling factor

    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.b = Parameter(torch.full((out_channels,1,1,1), fill_value=b))
        self.e = Parameter(torch.full((out_channels,1,1,1), fill_value=e))

    def qbits(self):
        return torch.sum(F.relu(self.b)) * functools.reduce(lambda a,b : a*b, self.weight.shape[1:])

    def qweight(self):
        weight = torch.mul(2**-self.e, self.weight) # scale weight
        b = F.relu(self.b) # bit depth cannot be negative
        clamped_weight = torch.clamp(weight, min=-2**(b-1), max=2**(b-1)-1) # ensure weight is within valid dynamic range
        return  torch.mul(2**self.e, ((clamped_weight.round() - clamped_weight).detach() + clamped_weight))

    def forward(self, x):
        return F.conv2d(x, self.qweight())
    
        