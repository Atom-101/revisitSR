import os
import glob
import random
import pickle

from ptsr.data import common
from ptsr.utils.utility import calc_psnr

import numpy as np
import imageio
from skimage.transform import resize  
import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, cfg, name='', train=True, benchmark=False):
        self.cfg = cfg
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = False
        self.scale = cfg.DATASET.DATA_SCALE
        self.idx_scale = 0
        self.psnrs = []
        
        self._set_filesystem(cfg.DATASET.DATA_DIR)
        if cfg.DATASET.DATA_EXT.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if cfg.DATASET.DATA_EXT.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr, list_lr = self._scan()
            self.images_hr = self._check_and_load(
                cfg.DATASET.DATA_EXT, list_hr, self._name_hrbin()
            )
            self.images_lr = [
                self._check_and_load(cfg.DATASET.DATA_EXT, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]
        else:
            if cfg.DATASET.DATA_EXT.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr
            elif cfg.DATASET.DATA_EXT.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )
                for s in self.scale:
                    os.makedirs(
                        os.path.join(
                            self.dir_lr.replace(self.apath, path_bin),
                            'X{}'.format(s)
                        ),
                        exist_ok=True
                    )
                
                self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(
                        cfg.DATASET.DATA_EXT, [h], b, verbose=True, load=False
                    )

                for i, ll in enumerate(list_lr):
                    for l in ll:
                        b = l.replace(self.apath, path_bin)
                        b = b.replace(self.ext[1], '.pt')
                        self.images_lr[i].append(b)
                        self._check_and_load(
                            cfg.DATASET.DATA_EXT, [l], b,  verbose=True, load=False
                        )

        if train:
            self.n_train_samples = cfg.SOLVER.ITERATION_TOTAL * cfg.SOLVER.SAMPLES_PER_BATCH
            n_patches = cfg.SOLVER.SAMPLES_PER_BATCH * cfg.SOLVER.TEST_EVERY
            n_images = len(cfg.DATASET.DATA_TRAIN) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]

        for i in range(self.begin, self.end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext[0]))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext[1])
                ))
        
        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            if ext.find('bin') >= 0:
                print('Bin pt file with name and image')
                b = [{
                    'name': os.path.splitext(os.path.basename(_l))[0],
                    'image': imageio.imread(_l)
                } for _l in l]
                with open(f, 'wb') as _f: pickle.dump(b, _f) 

                return b               
            else:
                print('Direct pt file without name or image')
                # import pdb
                # pdb.set_trace()
                b = imageio.imread(l[0])
                with open(f, 'wb') as _f: pickle.dump(b, _f)

            # return b

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.cfg.DATASET.CHANNELS)
        pair_t = common.np2Tensor(*pair, rgb_range=self.cfg.DATASET.RGB_RANGE)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return self.n_train_samples
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if not self.train:
            return idx

        idx = random.randrange(self.n_train_samples)
        return idx % len(self.images_hr)

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        if self.cfg.DATASET.DATA_EXT.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.cfg.DATASET.DATA_EXT == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.cfg.DATASET.DATA_EXT.find('sep') >= 0:
                # For each pt file, use 'image' to load it
                # with open(f_hr, 'rb') as _f: hr = pickle.load(_f)[0]['image']
                # with open(f_lr, 'rb') as _f: lr = pickle.load(_f)[0]['image']
                # For each pt file, directly load it
                with open(f_hr, 'rb') as _f: hr = pickle.load(_f)
                with open(f_lr, 'rb') as _f: lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        # wonr work for multiscale
        if self.train:
            lr_patch, hr_patch = common.get_patch(
                lr, hr,
                patch_size=self.cfg.DATASET.OUT_PATCH_SIZE,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            psnr = calc_psnr(torch.as_tensor(resize(lr_patch, hr_patch.shape[:2], order=3)).permute(2,0,1)[None], 
                    torch.as_tensor(hr_patch).permute(2,0,1)[None], 
                    scale, 
                    self.cfg.DATASET.RGB_RANGE
            )
            print(psnr)
            self.psnrs.append(psnr)
            if len(self.psnrs) >= 5000:
                with open('psnr_5000_file.txt', 'a') as f:
                    for psnr in self.psnrs:
                        print(psnr, file=f)
                raise ValueError()
            # if 15 < psnr < 20:
            #     import matplotlib.pyplot as plt
            #     fig,ax = plt.subplots(1,2)
            #     ax[0].imshow(hr_patch)
            #     ax[1].imshow(resize(lr_patch, hr_patch.shape[:2], order=3))
            #     # plt.imsave(f'psnr_{psnr:.2f}.png', hr_patch)
            #     fig.tight_layout()
            #     fig.savefig(f'psnr_plot_{psnr:.2f}.png')
            #     raise ValueError()
            if self.cfg.DATASET.REJECTION_SAMPLING.ENABLED and random.random() <= self.cfg.DATASET.REJECTION_SAMPLING.EPSILON:
                while calc_psnr(torch.as_tensor(resize(lr_patch, hr_patch.shape[:2], order=3)).permute(2,0,1), torch.as_tensor(hr_patch).permute(2,0,1), scale, self.cfg.DATASET.RGB_RANGE) > self.cfg.DATASET.REJECTION_SAMPLING.MAX_PSNR:
                    lr_patch, hr_patch = common.get_patch(
                        lr, hr,
                        patch_size=self.cfg.DATASET.OUT_PATCH_SIZE,
                        scale=scale,
                        multi=(len(self.scale) > 1),
                        input_large=self.input_large
                    )        
            
            if self.cfg.AUGMENT.ENABLED: lr_patch, hr_patch = common.augment(lr_patch, hr_patch)
        else:
            ih, iw = lr.shape[:2]
            hr_patch = hr[0:ih * scale, 0:iw * scale]
            lr_patch = lr
            
            psnr = calc_psnr(torch.as_tensor(resize(lr_patch, hr_patch.shape[:2], order=3)).permute(2,0,1)[None], 
                    torch.as_tensor(hr_patch).permute(2,0,1)[None], 
                    scale, 
                    self.cfg.DATASET.RGB_RANGE
            )
            print(psnr)
            self.psnrs.append(psnr)
            with open('psnr_test_file.txt', 'a') as f:
                print(psnr, file=f)
            if len(self.psnrs) >= 200:
                with open('psnr_test_file.txt', 'a') as f:
                    for psnr in self.psnrs:
                        print(psnr, file=f)
                raise ValueError()

        return lr_patch, hr_patch

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

