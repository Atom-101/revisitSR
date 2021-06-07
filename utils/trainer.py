import os
import math
import GPUtil
from decimal import Decimal

import torch
import torch.nn.utils as utils
from . import utility
from .solver import *
from tqdm import tqdm
from torch.cuda.amp import autocast


class Trainer():
    def __init__(self, args, cfg, loader, model, loss, device, ckp=None):
        self.args = args
        self.cfg = cfg
        self.scale = args.scale

        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.device = device

        if not args.test_only:
            self.loader_train = iter(loader.loader_train)
            self.optimizer = build_optimizer(cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
            self.error_last = 1e8
            self.iteration_total = self.cfg.SOLVER.ITERATION_TOTAL

    def train(self):
        self.model.train()

        timer = utility.timer()
        for i in range(self.iteration_total):

            lr, hr, _ = next(self.loader_train)
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.args.local_rank == 0 and i % 10 == 0:
                self.log_train(i, loss, timer)

            # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
            del lr, hr, sr, loss

            if (i+1) % self.cfg.SOLVER.ITERATION_VAL == 0:
                self.test(i)
                self.model.train()

    def log_train(self, i, loss, timer):
        lr = self.optimizer.param_groups[0]['lr']
        total_time = timer.toc()
        avg_itertime = total_time / (i+1)
        est_timeleft = avg_itertime * (self.iteration_total - i) / 3600
        print(
            "[Iteration %05d] Loss: %.5f, LR: %.5f, " % (i, loss.item(), lr)
            + "Iter time: %.4fs, Total time: %.2fh, Time Left %.2fh." % (
                avg_itertime, total_time / 3600, est_timeleft
            )
        )

        if i % 100 == 0 and torch.cuda.is_available():
            GPUtil.showUtilization(all=True)

    def test(self, epoch=None):
        self.model.eval()
        torch.set_grad_enabled(False)

        if self.ckp is not None:
            self.ckp.write_log('\nEvaluation:')
            self.ckp.add_log(
                torch.zeros(1, len(self.loader_test), len(self.scale))
            )

            timer_test = utility.timer()
            if self.args.save_results:
                self.ckp.begin_background()
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr = lr.to(self.device, non_blocking=True)
                        hr = hr.to(self.device, non_blocking=True)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(
                                d, filename[0], save_list, scale)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )

            self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
            self.ckp.write_log('Saving...')

            if self.args.save_results:
                self.ckp.end_background()

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(
                    best[1][0, 0] + 1 == epoch))

            self.ckp.write_log(
                'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )

        torch.set_grad_enabled(True)
