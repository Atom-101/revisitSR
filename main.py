import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from ptsr import model
from ptsr.data import Data
from ptsr.config import get_cfg_defaults
from ptsr.utils import template, utility, trainer

def init_seed(seed: int):
    import random
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Super Resolution')
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--distributed', action='store_true', help='distributed training')
    parser.add_argument('--local_rank', type=int, help='node rank for distributed training', default=None)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = get_cfg_defaults()

    manual_seed = 0 if args.local_rank is None else args.local_rank

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.local_rank==0 or args.local_rank is None:
        print(cfg)

    if args.distributed:
        assert torch.cuda.is_available(), \
            "Distributed training without GPUs is not supported!"
        dist.init_process_group("nccl", init_method='env://', world_size=cfg.SYSTEM.NUM_GPU, rank=args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)        
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.local_rank==0 or args.local_rank is None:
        print("Rank: {}. Device: {}".format(args.local_rank, device))

    init_seed(manual_seed)

    # These are the parameters used to initialize the process group
    # env_dict = {
    #     key: os.environ[key]
    #     for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    # }
    # print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    # dist.init_process_group(backend="nccl")
    # print(
    #     f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
    #     + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    # )

    # args.rank = int(os.environ["RANK"])
    # n = torch.cuda.device_count() // args.local_world_size
    # device_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))

    # torch.cuda.set_device(args.local_rank)
    # device = torch.device("cuda", args.local_rank)

    # print(
    #     f"[{os.getpid()}] rank = {dist.get_rank()} ({args.rank}), "
    #     + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    # )

    checkpoint = None
    # if args.local_rank == 0:
    checkpoint = utility.checkpoint(cfg)
    cudnn.enabled = True
    cudnn.benchmark = True

    _model, _loss = build_model_loss(cfg, args.local_rank, checkpoint, device)
    loader = Data(cfg)

    t = trainer.Trainer(cfg, args.local_rank, loader, _model, _loss, device, checkpoint)
    t.test() if cfg.SOLVER.TEST_ONLY else t.train()

    # Tear down the process group
    dist.destroy_process_group()


def build_model_loss(cfg, rank, checkpoint, device):
    _model = model.Model(cfg, checkpoint).to(device)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    find_unused_param = cfg.MODEL.STOCHASTIC_DEPTH

    _model = nn.parallel.DistributedDataParallel(
        _model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_param)

    _loss = None
    if not cfg.SOLVER.TEST_ONLY:
        _loss = nn.L1Loss().to(device)

    return _model, _loss


if __name__ == '__main__':
    main()
