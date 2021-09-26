from typing import List, Callable

import itertools
import numpy as np
import torch.nn as nn
from .common import *
from ._utils import conv3x3

block_dict = {
    'basicblock': PreActBasicBlock,
    'bottleneck': PreActBottleneck,
    'mbconv': MBConvBlock,
}


class ResidualGroup(nn.Module):
    def __init__(self, block_type: str, n_resblocks: int,
                 short_skip: bool = False, **kwargs):
        super().__init__()
        self.short_skip = short_skip

        assert block_type in block_dict
        blocks = [block_dict[block_type](**kwargs) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.body(x)
        if self.short_skip:
            res += x
        return res


class ISRNet(nn.Module):
    def __init__(self, n_resgroups: int, n_resblocks: int, planes: int, scale: int,
                 prob: List[float], block_type: str, short_skip: bool = False,
                 channels: int = 3, rgb_range: int = 255, act_mode: str = 'relu',
                 **kwargs):
        super().__init__()
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        modules_head = [conv3x3(channels, planes, bias=True)]
        modules_body = [ResidualGroup(
            block_type, n_resblocks, short_skip, planes=planes,
            act_mode=act_mode, prob=prob[i], **kwargs)
            for i in range(n_resgroups)]
        modules_tail = [
            Upsampler(scale, planes, act_mode),
            conv3x3(planes, channels, bias=True)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x  # long skip-connection

        x = self.tail(res)
        x = self.add_mean(x)

        return x


class Model(nn.Module):
    def __init__(self, cfg, ckp=None):
        super().__init__()
        self.scale = cfg.DATASET.DATA_SCALE[0]
        self.self_ensemble = cfg.MODEL.SELF_ENSEMBLE
        self.chop = cfg.DATASET.CHOP

        self.model = self.make_model(cfg)

        if ckp is not None:
            self.load(
                ckp.get_path('model'),
                pre_train=cfg.MODEL.PRE_TRAIN,
                resume=not cfg.SOLVER.ITERATION_RESTART,
                cpu=not bool(cfg.SYSTEM.NUM_GPU)
            )
            print(self.model, file=ckp.log_file)

    def make_model(self, cfg):
        n = cfg.MODEL.N_RESGROUPS
        options = {
            'n_resgroups': n,
            'n_resblocks': cfg.MODEL.N_RESBLOCKS,
            'planes': cfg.MODEL.PLANES,
            'scale': cfg.DATASET.DATA_SCALE[0],
            'block_type': cfg.MODEL.BLOCK_TYPE,
            'short_skip': cfg.MODEL.SHORT_SKIP,
            'channels': cfg.DATASET.CHANNELS,
            'rgb_range': cfg.DATASET.RGB_RANGE,
            'act_mode': cfg.MODEL.ACT_MODE,
            'stochastic_depth': cfg.MODEL.STOCHASTIC_DEPTH,
            'multFlag': cfg.MODEL.MULT_FLAG,
            'reduction': cfg.MODEL.SE_REDUCTION,  # SE block
            'zero_inti_residual': cfg.MODEL.ZERO_INIT_RESIDUAL,
        }

        # build a probability list for stochastic depth
        prob = cfg.MODEL.STOCHASTIC_DEPTH_PROB
        if prob is None:
            options['prob'] = [0.5] * n
        elif isinstance(prob, float):
            options['prob'] = [prob] * n
        elif isinstance(prob, list):
            assert len(prob) == 2
            n = cfg.MODEL.N_RESGROUPS
            temp = np.arange(n) / float(n-1)
            prob_list = prob[0] + temp * (prob[1] - prob[0])
            options['prob'] = list(prob_list)

        return ISRNet(**options)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url, model_dir=dir_model, **kwargs)
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs)

        if load_from and 'state_dict' in load_from.keys():
            self.model.load_state_dict(load_from['state_dict'], strict=False)

    def forward(self, x):
        if self.training:
            return self.model(x)

        # inference mode
        forward_func = self.forward_patch if self.chop else self.model.forward
        if self.self_ensemble:
            return self.forward_ensemble(x, forward_func=forward_func)

        return forward_func(x)

    def forward_patch(self, x, padding: int = 20, threshold: int = 160000):
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + padding, w_half + padding
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        n_samples = 2
        if (w_size * h_size) < threshold:  # smaller than the threshold
            sr_list = []
            for i in range(0, 4, n_samples):
                lr_batch = torch.cat(lr_list[i:(i + n_samples)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_samples, dim=0))
        else:
            sr_list = [
                self.forward_patch(patch, padding, threshold)
                for patch in lr_list]

        h, w = self.scale * h, self.scale * w
        h_half, w_half = self.scale * h_half, self.scale * w_half
        h_size, w_size = self.scale * h_size, self.scale * w_size
        padding *= self.scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_ensemble(self, x, forward_func: Callable):
        def _transform(data, xflip, yflip, transpose):
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
            return data

        outputs = []
        opts = itertools.product((False, True), (False, True), (False, True))
        for xflip, yflip, transpose in opts:
            data = x.clone()
            data = _transform(data, xflip, yflip, transpose)
            data = forward_func(data)
            outputs.append(_transform(data, xflip, yflip, transpose))

        return torch.stack(outputs, 0).mean(0)
