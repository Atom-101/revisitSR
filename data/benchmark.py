import os

from data import common
from data import srdata

import numpy as np

import torch
import torch.utils.data as data
import glob
class Benchmark(srdata.SRData):
    def __init__(self, cfg, name='', train=True, benchmark=True):
        super(Benchmark, self).__init__(
            cfg, name=name, train=train, benchmark=True
        )

    '''def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        if self.input_large:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubicL')
        else:
            self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('', '.png')'''
    def _set_filesystem(self, dir_data):

        self.apath = os.path.join(dir_data, 'sr_test')
        # base = "/n/pfister_lab2/Lab/sabdelmagid/new_folder/ICCV21/Benchmarks-Yulun/To_Salma/"
        print("self.scale = ",self.scale)
        self.dir_hr = os.path.join(self.apath, "HR/{}/x{}".format(self.name,self.scale[0]))

        self.dir_lr = os.path.join(self.apath,  "LRBI/{}/".format(self.name))
        self.ext = ('.png', '.png')
        
    def _scan(self):

        names_hr = sorted(
         glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            filename = filename.replace("_HR_x{}".format(self.scale[0]),"")
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                 self.dir_lr, 'x{}/{}_LRBI_x{}{}'.format(
                     s, filename, s, self.ext[1]
                 )
             ))
        # print(len(names_hr), len(names_lr[0]))
        return names_hr, names_lr