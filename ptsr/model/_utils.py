import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, 
            dilation: int = 1, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Swish(nn.Module):
    # An ordinary implementation of Swish function
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    # A memory-efficient implementation of Swish function
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def get_activation(activation: str = 'relu') -> nn.Module:
    """Get the specified activation layer. 
    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'``, 'efficient_swish'`` and ``'none'``. Default: ``'relu'``
    """
    assert activation in ["relu", "leaky_relu", "elu", "gelu",
                          "swish", "efficient_swish", "none"], \
        "Get unknown activation key {}".format(activation)
    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
        "efficient_swish": MemoryEfficientSwish(),
        "none": nn.Identity(),
    }
    return activation_dict[activation]