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


class Affine2d(nn.Module):
    def __init__(self, planes) -> None:
        super().__init__()
        self.weight = Parameter(torch.ones(1, planes, 1, 1))
        self.bias = Parameter(torch.zeros(1, planes, 1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(Affine2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(Affine2d(n_feats))
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


class PreActBase(nn.Module):
    def __init__(self, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__()
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

        # This block is skipped during training
        for param in self.parameters():
            param.requires_grad = False
        return identity

    def _forward_test(self, x, identity):
        res = self._forward_res(x)
        if self.multFlag:
            res *= self.prob

        return identity + res


class PreActBasicBlock(PreActBase):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes)
        self.conv1 = conv3x3(planes, planes)

        self.aff2 = Affine2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.relu = nn.ReLU(inplace=True)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aff1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.aff2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x


class PreActBottleneck(PreActBase):
    def __init__(self, planes: int, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__(stochastic_depth, prob, multFlag)
        self.aff1 = Affine2d(planes)
        self.conv1 = conv1x1(planes, planes)

        self.aff2 = Affine2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.aff3 = Affine2d(planes)
        self.conv3 = conv1x1(planes, planes)

        self.relu = nn.ReLU(inplace=True)

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


class DropPath(nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.p==0):
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (1-self.p) + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(1-self.p) * random_tensor
        return output


class SEBlock(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 24) -> None:
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y

class MBConvN(PreActBase):
    def __init__(self, in_planes: int, out_planes: int, expansion_factor: int, kernel_size: int = 3,
                 stride: int = 1, skip_conn: bool = True, r: int = 24, p: float = 0, stochastic_depth: bool = False,
                 prob: float = 1.0, multFlag: bool = True) -> None:
        super().__init__(stochastic_depth, prob, multFlag)

        padding = (kernel_size-1)//2
        exp_planes = in_planes*expansion_factor
        self.skip_connection = (in_planes==out_planes) and (stride==1) and skip_conn
        
        #Pointwise Expand Convolution
        self.pw_conv1 = conv1x1(in_planes, exp_planes)
        self.bn1 = Affine2d(exp_planes)
        self.act1 = nn.ReLU(inplace=True)

        #Depthwise Convolution
        self.dw_conv = nn.Conv2d(exp_planes, exp_planes, kernel_size, stride, padding, groups=exp_planes)
        self.bn2 = Affine2d(exp_planes)
        self.act2 = nn.ReLU(inplace=True)

        #Squeeze-Excitation Layer
        self.se = SEBlock(exp_planes, reduction=r)

        #Pointwise Reduce Convolution
        self.pw_conv2 = conv1x1(exp_planes, out_planes)
        self.bn3 = Affine2d(out_planes) #No Activation

        self.droppath = DropPath(p)

    def _forward_res(self, x: torch.Tensor) -> torch.Tensor:
        residual= x

        x = self.pw_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dw_conv(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.se(x)

        x = self.pw_conv2(x)
        x = self.bn3(x)

        if self.skip_connection:
            x = self.droppath(x)
            x += residual

        return x
