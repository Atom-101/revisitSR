import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

############################
# Residual Blocks (Updated)
############################


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Affine2d(nn.Module):
    def __init__(self, planes) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(1, planes, 1, 1))
        self.bias = Parameter(torch.zeros(1, planes, 1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


class PreActBasicBlock(nn.Module):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__()
        self.aff1 = Affine2d(planes)
        self.conv1 = conv3x3(planes, planes)

        self.aff2 = Affine2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.relu = nn.ReLU(inplace=True)

        self.sd = stochastic_depth
        if stochastic_depth:
            self.prob = prob
            self.multFlag = multFlag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        if self.training:
            return self._forward_train(x, identity)

        return self._forward_test(x, identity)

    def _forward_train(self, x, identity):
        if not self.sd or torch.rand(1) < self.prob:
            for param in self.parameters():
                param.requires_grad = True
            res = self._forward_res(x)
            return identity + res

        # This block is skipped durint training
        for param in self.body.parameters():
            param.requires_grad = False
        return identity

    def _forward_test(self, x, identity):
        res = self._forward_res(x)
        if self.multFlag:
            res *= self.prob

        return identity + res

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x


class PreActBottleneck(nn.Module):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__()
        self.aff1 = Affine2d(planes)
        self.conv1 = conv1x1(planes, planes)

        self.aff2 = Affine2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.aff3 = Affine2d(planes)
        self.conv3 = conv1x1(planes, planes)

        self.relu = nn.ReLU(inplace=True)

        self.sd = stochastic_depth
        if stochastic_depth:
            self.prob = prob
            self.multFlag = multFlag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        if self.training:
            return self._forward_train(x, identity)

        return self._forward_test(x, identity)

    def _forward_train(self, x, identity):
        if not self.sd or torch.rand(1) < self.prob:
            for param in self.parameters():
                param.requires_grad = True
            res = self._forward_res(x)
            return identity + res

        # This block is skipped durint training
        for param in self.body.parameters():
            param.requires_grad = False
        return identity

    def _forward_test(self, x, identity):
        res = self._forward_res(x)
        if self.multFlag:
            res *= self.prob

        return identity + res

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.aff3(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x
