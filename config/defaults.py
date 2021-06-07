from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

# Mixed-precision training/inference
_C.MODEL.MIXED_PRECESION = False

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()

# Solver type needs to be one of 'sgd', 'adam'
_C.SOLVER.TYPE = 'sgd'
# Specify the learning rate scheduler.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

# Save a checkpoint after every this number of iterations.
_C.SOLVER.ITERATION_SAVE = 5000
_C.SOLVER.ITERATION_TOTAL = 1000000
_C.SOLVER.ITERATION_VAL = 1000

# Whether or not to restart training from iteration 0 regardless
# of the 'iteration' key in the checkpoint file. This option only
# works when a pretrained checkpoint is loaded (default: False).
_C.SOLVER.ITERATION_RESTART = False

_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.MOMENTUM = 0.9  # SGD
_C.SOLVER.BETAS = (0.9, 0.999)  # ADAM

# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

# The iteration number to decrease learning rate by GAMMA
_C.SOLVER.GAMMA = 0.1

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
_C.SOLVER.SWA.START_ITER = 90000
_C.SOLVER.SWA.MERGE_ITER = 10
_C.SOLVER.SWA.BN_UPDATE_ITER = 2000


def get_cfg_defaults():
    r"""Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
