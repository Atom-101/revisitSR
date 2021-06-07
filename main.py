import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import data
import model
from config import get_cfg_defaults
from utils import template, utility, trainer


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Super Resolution')

    parser.add_argument('--debug', action='store_true',
                        help='Enables debug mode')
    parser.add_argument('--template', default='.',
                        help='You can set various templates in option.py')

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=4,
                        help='number of threads (per process) for data loading')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    # Data specifications
    parser.add_argument('--dir_data', type=str, default='/n/pfister_lab2/Lab/vcg_natural/SR/BIX2X3X4',
                        help='dataset directory')
    parser.add_argument('--dir_demo', type=str, default='../test',
                        help='demo image directory')
    parser.add_argument('--data_train', type=str, default='DF2K',
                        help='train dataset name')
    parser.add_argument('--data_test', type=str, default='DF2K',
                        help='test dataset name')
    parser.add_argument('--data_range', type=str, default='1-3550/3551-3555',
                        help='train/test data range')
    parser.add_argument('--ext', type=str, default='bin',
                        help='dataset file extension, img, sep, bin. In cluster, use bin')
    parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')
    parser.add_argument('--patch_size', type=int, default=192,
                        help='output patch size')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')

    # Model specifications
    parser.add_argument('--model', default='EDSR',
                        help='model name')

    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--pre_train', type=str, default='',
                        help='pre-trained model directory')
    parser.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--n_resblocks', type=int, default=16,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1.0,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--dilation', action='store_true',
                        help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')

    # Option for Residual dense network (RDN)
    parser.add_argument('--G0', type=int, default=64,
                        help='default number of filters. (Use in RDN)')
    parser.add_argument('--RDNkSize', type=int, default=3,
                        help='default kernel size. (Use in RDN)')
    parser.add_argument('--RDNconfig', type=str, default='B',
                        help='parameters config of RDN. (Use in RDN)')

    # Option for Residual channel attention network (RCAN)
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    # Training specifications
    parser.add_argument('--reset', action='store_true',
                        help='reset the training')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--iterations', type=int, default=150000,
                        help='number of iterations to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--split_batch', type=int, default=1,
                        help='split the batch into smaller chunks')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='use self-ensemble method for test')
    parser.add_argument('--test_only', action='store_true',
                        help='set this option to test the model')
    parser.add_argument('--gan_k', type=int, default=1,
                        help='k value for adversarial loss')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=200,
                        help='learning rate decay per N iterations')
    parser.add_argument('--decay_type', type=str, default='step',
                        help='learning rate decay type')
    parser.add_argument('--decay', type=str, default='200-400-600-800-1000-1200-1400-1600-1800-2000',
                        help='learning rate decay type, multiple_step, 200-400-600-800-1000-1200-1400-1600-1800-2000')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                        help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--gclip', type=float, default=0,
                        help='gradient clipping threshold (0 = no clipping)')

    # Loss specifications
    parser.add_argument('--loss', type=str, default='1*L1',
                        help='loss function configuration')
    parser.add_argument('--skip_threshold', type=float, default='1e8',
                        help='skipping batch that has large error')

    # Log specifications
    parser.add_argument('--save', type=str, default='test',
                        help='file name to save')
    parser.add_argument('--load', type=str, default='',
                        help='file name to load')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume from specific checkpoint')
    parser.add_argument('--save_models', action='store_true',
                        help='save all intermediate models')
    parser.add_argument('--print_every', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_results', action='store_true',
                        help='save output results')
    parser.add_argument('--save_gt', action='store_true',
                        help='save low-resolution and high-resolution images together')

    args = parser.parse_args()
    template.set_template(args)

    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    args.data_train = args.data_train.split('+')
    args.data_test = args.data_test.split('+')

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    return args


def main():
    args = get_args()
    cfg = get_cfg_defaults()

    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    args.rank = int(os.environ["RANK"])
    n = torch.cuda.device_count() // args.local_world_size
    device_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()} ({args.rank}), "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    )

    checkpoint = None
    if args.rank == 0:
        checkpoint = utility.checkpoint(args, cfg)

    cudnn.enabled = True
    cudnn.benchmark = True

    # set seeds based on global rank
    np.random.seed(args.rank)
    torch.manual_seed(args.rank)

    _model, _loss = build_model_loss(args, checkpoint, device, device_ids)
    loader = data.Data(args, cfg)

    t = trainer.Trainer(args, cfg, loader, _model, _loss, device, checkpoint)
    t.test() if args.test_only else t.train()

    # Tear down the process group
    dist.destroy_process_group()


def build_model_loss(args, checkpoint, device, device_ids):
    _model = model.Model(args, checkpoint).to(device)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    _model = nn.parallel.DistributedDataParallel(
        _model, device_ids=device_ids, output_device=args.local_rank)

    _loss = None
    if not args.test_only:
        _loss = nn.L1Loss().to(device)

    return _model, _loss


if __name__ == '__main__':
    main()
