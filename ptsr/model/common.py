import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ._utils import conv3x3, conv1x1, get_activation


class ResidualBase(nn.Module):
    def __init__(self, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__()
        self.sd = stochastic_depth
        if stochastic_depth:
            self.prob = prob
            self.multFlag = multFlag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        return self._forward_train(x, identity) if self.training \
            else self._forward_test(x, identity)

    def _forward_train(self, x, identity) -> torch.Tensor:
        if not self.sd or torch.rand(1) < self.prob:
            for param in self.parameters():
                param.requires_grad = True
            res = self._forward_res(x)
            return identity + res

        # This block is skipped during training
        for param in self.parameters():
            param.requires_grad = False
        return identity

    def _forward_test(self, x, identity) -> torch.Tensor:
        res = self._forward_res(x)
        if self.sd and self.multFlag:
            res *= self.prob

        return identity + res

    def _forward_res(self, x) -> torch.Tensor:
        # Residual forward function should be
        # defined in child classes.
        return 0


class PreActBasicBlock(ResidualBase):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 act_mode: str = 'relu', prob: float = 1.0,
                 multFlag: bool = True, **_):
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes)
        self.conv1 = conv3x3(planes, planes)

        self.aff2 = Affine2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.act = get_activation(act_mode)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)

        return x


class PreActBottleneck(ResidualBase):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 act_mode: str = 'relu', prob: float = 1.0,
                 multFlag: bool = True, **_):
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes)
        self.conv1 = conv1x1(planes, planes)

        self.aff2 = Affine2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.aff3 = Affine2d(planes)
        self.conv3 = conv1x1(planes, planes)

        self.act = get_activation(act_mode)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.aff3(x)
        x = self.act(x)
        x = self.conv3(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, planes: int, reduction: int = 8, act_mode: str = 'relu'):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(planes, planes // reduction, kernel_size=1),
            get_activation(act_mode),
            nn.Conv2d(planes // reduction, planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class MBConvBlock(ResidualBase):
    def __init__(self, planes: int, stochastic_depth: bool = False, act_mode: str = 'relu',
                 prob: float = 1.0, multFlag: bool = True, reduction: int = 8) -> None:
        super().__init__(stochastic_depth, prob, multFlag)

        self.conv1 = conv1x1(planes, planes)
        self.aff1 = Affine2d(planes)

        self.conv2 = conv3x3(planes, planes, groups=planes)  # depth-wise
        self.aff2 = Affine2d(planes)
        self.se = SEBlock(planes, reduction, act_mode)

        self.conv3 = conv1x1(planes, planes)
        self.aff3 = Affine2d(planes)

        self.act = get_activation(act_mode)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.aff1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.aff2(x)
        x = self.act(x)

        x = self.se(x)

        x = self.conv3(x)
        x = self.aff3(x)  # no activation

        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class Affine2d(nn.Module):
    def __init__(self, planes) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(1, planes, 1, 1))
        self.bias = Parameter(torch.zeros(1, planes, 1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, planes: int, act_mode: str = 'relu'):
        m = []
        if (scale & (scale - 1)) == 0:  # is power of 2
            for _ in range(int(math.log(scale, 2))):
                m.append(conv3x3(planes, 4 * planes))
                m.append(nn.PixelShuffle(2))
                m.append(Affine2d(planes))
                m.append(get_activation(act_mode))

        elif scale == 3:
            m.append(conv3x3(planes, 9 * planes))
            m.append(nn.PixelShuffle(3))
            m.append(Affine2d(planes))
            m.append(get_activation(act_mode))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
