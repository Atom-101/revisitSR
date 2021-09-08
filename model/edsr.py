from model import common

import torch.nn as nn

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(cfg, parent=False):
    return EDSR(cfg)

class EDSR(nn.Module):
    def __init__(self, cfg, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = cfg.MODEL.N_RESBLOCKS
        resblock = cfg.MODEL.RESBLOCK
        n_feats = cfg.MODEL.N_FEATS
        stochastic_depth = cfg.MODEL.STOCHASTIC_DEPTH
        multflag = cfg.MODEL.MULTFLAG
        p_resblock = cfg.MODEL.P_RESBLOCK
        kernel_size = 3 
        scale = cfg.DATASET.DATA_SCALE[0]
        act_list = [nn.ReLU(inplace=True), nn.LeakyReLU(inplace=True), nn.SiLU(inplace=True)]
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(cfg.DATASET.RGB_RANGE)
        self.add_mean = common.MeanShift(cfg.DATASET.RGB_RANGE, sign=1)

        self.act = act_list[cfg.MODEL.ACT]

        # define head module
        m_head = [conv(cfg.DATASET.CHANNELS, n_feats, kernel_size)]

        # define body module
        if resblock == 'mbconv':
            m_body = [
                common.MBConvN(
                    n_feats, n_feats, cfg.MODEL.EXPANSION, cfg.MODEL.KERNEL_SIZE, r = cfg.MODEL.SE_REDUCTION,
                    stochastic_depth=stochastic_depth, act_layer=self.act, prob=p_resblock, multFlag=multflag
                ) for _ in range(n_resblocks)
            ]
            m_body.append(conv(n_feats, n_feats, kernel_size))

        else:
            if resblock == 'basic':
                ResBlock=common.PreActBasicBlock
            elif resblock == 'bottleneck':
                ResBlock=common.PreActBottleneck

            m_body = [
                ResBlock(
                    n_feats, stochastic_depth=stochastic_depth, act_layer=self.act, prob=p_resblock, multFlag=multflag
                ) for _ in range(n_resblocks)
            ]
            m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, cfg.DATASET.CHANNELS, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

