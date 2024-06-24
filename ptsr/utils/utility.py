import torch.optim.lr_scheduler as lrs
import torch.optim as optim
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart:
            self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, cfg):
        # self.args = args
        self.cfg = cfg
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.datatest = []

        if (self.cfg.SOLVER.TEST_EVERY and not self.cfg.SOLVER.TEST_ONLY):
            self.datatest = self.cfg.DATASET.DATA_VAL
        elif (self.cfg.SOLVER.TEST_ONLY):
            self.datatest = self.cfg.DATASET.DATA_TEST

        if not cfg.LOG.LOAD:
            if not cfg.LOG.SAVE:
                cfg.LOG.SAVE = now
            self.dir = os.path.join('..', 'experiment', cfg.LOG.SAVE)
        else:
            self.dir = os.path.join('..', 'experiment', cfg.LOG.LOAD)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)*self.cfg.SOLVER.TEST_EVERY))
            else:
                cfg.LOG.LOAD = ''

        if cfg.SOLVER.ITERATION_RESTART:
            os.system('rm -rf ' + self.dir)
            cfg.LOG.LOAD = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in self.datatest:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            print(cfg, file=f)
            # for arg in vars(args):
            #     f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, iteration, is_best=False):
        self.save_model(self.get_path('model'), trainer, iteration, is_best)
        self.plot_psnr(iteration)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def save_model(self, apath, trainer, iteration: int, is_best: bool = False):
        save_dirs = [os.path.join(apath, 'model_latest.pth.tar')]
        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pth.tar'))

        state = {'iteration': iteration + 1,
                 'state_dict': trainer.model.module.model.state_dict(),  # DP, DDP
                 'optimizer': trainer.optimizer.state_dict(),
                 'lr_scheduler': trainer.lr_scheduler.state_dict()}

        for filename in save_dirs:
            torch.save(state, filename)

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, iteration):
        intervel = self.cfg.SOLVER.TEST_EVERY
        num_points = (iteration + 1) // intervel
        axis = list(range(1, num_points+1))
        for idx_data, d in enumerate(self.datatest):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.cfg.DATASET.DATA_SCALE):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None:
                        break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]

        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.cfg.LOG.SAVE_RESULTS:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.cfg.DATASET.RGB_RANGE)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    # if diff.size(1) > 1:
    #     gray_coeffs = [65.738, 129.057, 25.064]
    #     convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
    #     diff = diff.mul(convert).sum(dim=1)

    shave = scale
    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    
    if mse == 0:  # PSNR have no importance.
        return 100

    return -10 * math.log10(mse)
