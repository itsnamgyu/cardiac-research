"""
Parameters for final module
"""

BATCH_SIZE = 32
# save intermediate weights
SAVE_ALL_WEIGHTS = False
# interval for saving intermediate weights (in epochs)
T = 10
# freeze depth index
DEPTH_INDEX = 0
# multiplier for out_of_myocardial (OAP, OBS) slices
BALANCE = 5
LEARNING_RATE = 0.001
EPOCHS = 5
# Fraction of dataset to use during train/validation. Set to 0.05-0.10 for sanity
# checking. Set to None to use entire dataset.
SAMPLE = 0.05
# model
MODEL_KEY = 'mobilenet_a25'

# Choices
#'vgg16',
#'mobilenet_a25',
#'mobilenet_v2_a35',
#'resnet50_v2',
#'inception_v3',
#'inception_resnet_v2',
#'densenet_121',
#'nasnet_mobile',
#'xception',
#'baseline_model_v1'

# use_multiprocessing, workers arguments for fit/predict functions
USE_MULTIPROCESSING = False
MULTIPROCESSING_WORKERS = 16
