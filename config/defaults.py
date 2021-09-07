from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.DEBUG = False
# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.NUM_CPU = 4
_C.SYSTEM.NUM_GPU = 1
_C.SYSTEM.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Mixed-precision training/inference
_C.MODEL.MIXED_PRECESION = False

_C.MODEL.NAME = 'RCAN'
_C.MODEL.ACT = 'relu'
_C.MODEL.PRE_TRAIN = ''
_C.MODEL.EXTEND = '.'
_C.MODEL.RESBLOCK = 'basic'
_C.MODEL.N_RESBLOCKS = 16
_C.MODEL.N_FEATS = 64
_C.MODEL.STOCHASTIC_DEPTH = False
_C.MODEL.P_RESBLOCK = 1.0
_C.MODEL.MULTFLAG = True
_C.MODEL.EXPANSION = 1
_C.MODEL.KERNEL_SIZE = 3
_C.MODEL.RES_SCALE = 1.0
_C.MODEL.SHIFT_MEAN = True
_C.MODEL.DILATION = False
_C.MODEL.SELF_ENSEMBLE = False
_C.MODEL.SE_REDUCTION = 24

# RDN specific
_C.MODEL.G0 = 64
_C.MODEL.RDN_K_SIZE = 3
_C.MODEL.RDN_CONFIG = 'B'

# RCAN specific
_C.MODEL.N_RESGROUPS = 10
_C.MODEL.REDUCTION = 16

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATA_DIR = '/n/pfister_lab2/Lab/vcg_natural/SR/BIX2X3X4'
_C.DATASET.DEMO_DIR = '../test'
_C.DATASET.DATA_TRAIN = ['DF2K']
_C.DATASET.DATA_VAL = ['DF2K']
_C.DATASET.DATA_TEST = ['DF2K', 'Set5', 'Set14C', 'B100', 'Urban100', 'Manga109']
_C.DATASET.DATA_RANGE = [(1,3550), (3551,3555)]
_C.DATASET.DATA_EXT = 'img' #'bin', 'sep' or 'img'
_C.DATASET.DATA_SCALE = [4]
_C.DATASET.OUT_PATCH_SIZE = 192
_C.DATASET.RGB_RANGE = 255
_C.DATASET.CHANNELS = 3
_C.DATASET.CHOP = False


# -----------------------------------------------------------------------------
# Augment
# -----------------------------------------------------------------------------
_C.AUGMENT = CN({"ENABLED": False})


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

# Solver type needs to be one of 'sgd', 'adam'
_C.SOLVER.TYPE = 'adam'
# Specify the learning rate scheduler.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

# Save a checkpoint after every this number of iterations.
_C.SOLVER.ITERATION_SAVE = 5000
_C.SOLVER.ITERATION_TOTAL = 160000
_C.SOLVER.TEST_EVERY = 1000

# Whether or not to restart training from iteration 0 regardless
# of the 'iteration' key in the checkpoint file. This option only
# works when a pretrained checkpoint is loaded (default: False).
_C.SOLVER.ITERATION_RESTART = True

_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.MOMENTUM = 0.9  # SGD
_C.SOLVER.BETAS = (0.9, 0.999)  # ADAM

# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY = 0 #0.0001
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

# The iteration number to decrease learning rate by GAMMA
_C.SOLVER.GAMMA = 0.5

# should be a tuple like (30000,)
_C.SOLVER.STEPS = (30000, 35000)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000

_C.SOLVER.WARMUP_ITERS = 10000

_C.SOLVER.WARMUP_METHOD = "linear"

# Number of samples per GPU. If we have 8 GPUs and SAMPLES_PER_BATCH = 8,
# then each GPU will see 2 samples and the effective batch size is 64.
_C.SOLVER.SAMPLES_PER_BATCH = 16

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Stochastic Weight Averaging
_C.SOLVER.SWA = CN({"ENABLED": False})
_C.SOLVER.SWA.LR_FACTOR = 0.05
_C.SOLVER.SWA.START_ITER = 150000
_C.SOLVER.SWA.MERGE_ITER = 50
_C.SOLVER.SWA.BN_UPDATE_ITER = 2000

_C.SOLVER.SELF_ENSEMBLE = False
_C.SOLVER.TEST_ONLY = False
_C.SOLVER.GAN_K = 1

_C.SOLVER.LOSS = CN()
_C.SOLVER.LOSS.CONFIG = '1*L1'
_C.SOLVER.LOSS.SKIP_THRES = 1e8


# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.SAVE = 'test'
_C.LOG.LOAD = ''
_C.LOG.RESUME = 0
_C.LOG.SAVE_MODELS = False
_C.LOG.PRINT_EVERY = 100
_C.LOG.SAVE_RESULTS = True
_C.LOG.SAVE_GT = True


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
