from ptsr.data import srdata

class SR291(srdata.SRData):
    def __init__(self, cfg, name='SR291', train=True, benchmark=False):
        super(SR291, self).__init__(cfg, name=name)

