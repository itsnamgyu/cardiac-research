"""
Parameters for final module
"""

BATCH_SIZE = 32
EPOCHS = 50
# Multiplier for out_of_myocardial (OAP, OBS) slices
BALANCE = 5

# Fraction of dataset to use during train/validation. Set to 0.05-0.10 for sanity
# checking. Set to None to use entire dataset.
SAMPLE = None
# SAMPLE = 0.05

# Save intermediate weights
SAVE_ALL_WEIGHTS = False
# Interval for saving intermediate weights (in epochs)
T = 10

# Depth index for layerwise fine-tuning. Exact layers are specified
# per model in core.fine_model
DEPTH_INDEX = 0
# DEPTH_INDEX = 1

# Learning rate
LEARNING_RATE = 0.001

# Model
MODEL_KEY = 'mobilenet_a25'
#'vgg16'
#'mobilenet_a25'
#'mobilenet_v2_a35'
#'resnet50_v2'
#'inception_v3'
#'inception_resnet_v2'
#'densenet_121'
#'nasnet_mobile'
#'xception'
#'baseline_model_v1'
#'cbt_large_t'
#'cbt_large_w'
#'cbt_small'
#'cbt_tiny'

# use_multiprocessing, workers arguments for fit/predict functions
USE_MULTIPROCESSING = False
MULTIPROCESSING_WORKERS = 16
