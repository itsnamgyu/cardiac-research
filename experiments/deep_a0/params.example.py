"""
Parameters for main module
"""

BATCH_SIZE = 32
# number of folds
K = 5
# save intermediate weights
SAVE_ALL_WEIGHTS = False
# interval for saving intermediate weights (in epochs)
T = 10
# multiplier for out_of_myocardial (OAP, OBS) slices
BALANCE = 5
LEARNING_RATES = [0.01, 0.001, 0.0001]
EPOCHS = 50
# Fraction of dataset to use during train/validation. Set to 0.05-0.10 for sanity
# checking. Set to None to use entire dataset.
SAMPLE = None
# SAMPLE = 0.05
# seed for k-fold split
K_SPLIT_SEED = 1
# models to train
MODEL_KEYS = [
    #'vgg16',
    'mobilenet_a25',
    #'mobilenet_v2_a35',
    #'resnet50_v2',
    #'inception_v3',
    #'inception_resnet_v2',
    #'densenet_121',
    #'nasnet_mobile',
    #'xception',
    #'baseline_model_v1'
]
# Setup the module to train for one specific learning rate (0...2)
# LR_INDEX = None: train on all learning rates
LR_INDEX = None
# Setup the module to train for one specific fold (0...4)
# FOLD_INDEX = None: train on all folds
FOLD_INDEX = None
# use_multiprocessing, workers arguments for fit/predict functions
USE_MULTIPROCESSING = False
MULTIPROCESSING_WORKERS = 16
