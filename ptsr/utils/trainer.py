import os
import math
import GPUtil
from decimal import Decimal

import torch
import torch.nn.utils as utils
from . import utility
from .solver import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR


class Trainer():
    def __init__(self, cfg, rank, loader, model, loss, device, ckp=None):
        self.cfg = cfg
        self.rank = rank
        self.scale = cfg.DATASET.DATA_SCALE

        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = model
        self.loss = loss
        self.device = device

        if not cfg.SOLVER.TEST_ONLY:
            self.loader_train = iter(loader.loader_train)
            self.optimizer = build_optimizer(cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer)
            self.error_last = 1e8
            self.iteration_total = self.cfg.SOLVER.ITERATION_TOTAL
        
        self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None

        if self.cfg.SOLVER.SWA.ENABLED and not self.cfg.SOLVER.TEST_ONLY:
            self.swa_model = AveragedModel(self.model).to(self.device)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr =self.cfg.SOLVER.SWA.LR_FACTOR)

    def train(self):
        self.model.train()
        timer = utility.timer()
        for i in range(self.iteration_total):

            lr, hr, _ = next(self.loader_train)
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                sr = self.model(lr)
                loss = self.loss(sr, hr)

            if self.cfg.MODEL.MIXED_PRECESION:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.maybe_update_swa_model(i)

            if (self.rank is None or self.rank == 0) and i % 10 == 0:
                self.log_train(i, loss, timer)

            # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
            del lr, hr, sr, loss

            if (i+1) % self.cfg.SOLVER.TEST_EVERY == 0:
                self.test(i+1)
                self.model.train()
        self.maybe_save_swa_model()

    def log_train(self, i, loss, timer):
        lr = self.optimizer.param_groups[0]['lr']
        total_time = timer.toc()
        avg_itertime = total_time / (i+1)
        est_timeleft = avg_itertime * (self.iteration_total - i) / 3600
        print(
            "[Iteration %05d] Loss: %.5f, LR: %.5f, " % (i+1, loss.item(), lr)
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
            if self.cfg.LOG.SAVE_RESULTS:
                self.ckp.begin_background()
            for idx_data, d in enumerate(self.loader_test):
                for idx_scale, scale in enumerate(self.scale):
                    d.dataset.set_scale(idx_scale)
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr = lr.to(self.device, non_blocking=True)
                        hr = hr.to(self.device, non_blocking=True)
                        sr = self.model(lr)
                        sr = utility.quantize(sr, self.cfg.DATASET.RGB_RANGE)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.cfg.DATASET.RGB_RANGE, dataset=d
                        )
                        if self.cfg.LOG.SAVE_GT:
                            save_list.extend([lr, hr])

                        if self.cfg.LOG.SAVE_RESULTS:
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
                            (best[1][idx_data, idx_scale] + 1)*self.cfg.SOLVER.TEST_EVERY
                        )
                    )

            self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
            self.ckp.write_log('Saving...')

            if self.cfg.LOG.SAVE_RESULTS:
                self.ckp.end_background()

            if not self.cfg.SOLVER.TEST_ONLY:
                self.ckp.save(self, epoch, is_best=(
                    (best[1][0, 0] + 1)*self.cfg.SOLVER.TEST_EVERY == epoch))

            self.ckp.write_log(
                'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
            )

        torch.set_grad_enabled(True)

    def maybe_save_swa_model(self):
        if not hasattr(self, 'swa_model'):
            return

        if self.cfg.MODEL.NORM_MODE in ['bn', 'sync_bn']:  # update bn statistics
            for _ in range(self.cfg.SOLVER.SWA.BN_UPDATE_ITER):
                sample = next(self.loader_train)
                volume = sample.out_input
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    _ = self.swa_model(volume)

        # save swa model
        if self.rank is None or self.rank==0:
            print("Save SWA model checkpoint.")
            filename = os.path.join(self.ckp.get_path('model'), 'model_swa.pth.tar')
            state = {'state_dict': self.swa_model.module.state_dict()}
            torch.save(state, filename)

        # maybe run test with swa model?

    def maybe_update_swa_model(self, iter_total):
        if not hasattr(self, 'swa_model'):
            self.lr_scheduler.step()
            return

        swa_start = self.cfg.SOLVER.SWA.START_ITER
        swa_merge = self.cfg.SOLVER.SWA.MERGE_ITER
        if iter_total >= swa_start:
            if iter_total % swa_merge == 0:
                self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        else:
            self.lr_scheduler.step()

