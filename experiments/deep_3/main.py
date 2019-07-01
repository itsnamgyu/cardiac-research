#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
mpl.use(
    'Agg'
)  # don't display mpl windows (will cause error in non-gui environment)

import main

from collections import defaultdict
import math
import os
import shutil
import pandas as pd
from bayes_opt import BayesianOptimization
import keras

import core.history as ch
import core.fine_model as cm
from core.fine_model import FineModel

import cr_interface as cri
import keras_utils as ku

import results
import analysis

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
# experiment index to track saved model weights, training history etc.
# iterate this index for each run (make sure to keep track of this index)
EXP = 7
# whether to sample 10% of all slices (for sanity checking purposes)
SAMPLE = False
# seed for k-fold split
K_SPLIT_SEED = 1
# models to train
MODEL_KEYS = [
    #'xception',
    'mobileneta25',
    #'mobilenetv2a35',
    #'vgg16',
    #'resnet50v2',
    #'inception_v3',
    #'inception_resnet_v2',
    #'densenet121',
    #'nasnet_mobile',
]

# Setup the module to train for one specific learning rate (0...2)
# LR_INDEX = None: train on all learning rates
LR_INDEX = 0

# Setup the module to train for one specific fold (0...4)
# FOLD_INDEX = None: train on all folds
FOLD_INDEX = 0

# use_multiprocessing, workers arguments for fit/predict functions
USE_MULTIPROCESSING = False
MULTIPROCESSING_WORKERS = 16


def run_by_fold(fm,
                depth_index,
                lr_index,
                epochs,
                train_gens,
                val_gens,
                test_gen,
                fold_index=None):
    """
    Train the model (frozen at some depth) for all five folds OR a specific fold


    Saves intermediate models with the following KEYS: [load weights via fm.load_weights(KEY)]
    EXP01_D01_L03_F01:
    Fully trained model for the 1st freeze depth, 3rd learning rate, fold 1
    EXP01_D01_L03_F01_E025:
    Partially trained model for the 1st freeze depth, 3rd learning rate, fold 1, until the 25th epoch

    Saves training history with the following KEYS: [get data via ch.get_history(model_name, KEY)]
    EXP01_D01_L03_F01:
    Training history for the 1st freeze depth, 3rd learning rate, fold 1


    :param fm:
    FineModel to train, i.e., the base network to train on

    :param depth_index:
    The INDEX of the "freeze depth" for the given FineModel

    :param lr_index:
    The INDEX of the learning rate, i.e., lr = LEARNING_RATES[lr_index]

    :param epochs:
    Number of epochs to train. MUST BE MULTIPLE OF 5.

    :param train_gens
    List of train ImageDataGenerators for each fold

    :param val_gens
    List of validation ImageDataGenerators for each fold

    :param val_gens
    Test ImageDataGenerator for each fold

    :param fold_index
    If specified, will only run the specific fold index
    """
    _exp_key = 'EXP{:02}'.format(EXP)
    _depth_key = _exp_key + '_D{:02}'
    _fold_key = _depth_key + '_L{:02}_F{:02}'
    _epoch_key = _fold_key + '_E{:03}'

    lr = LEARNING_RATES[lr_index]
    loss_list = []
    acc_list = []

    folds = range(K)
    if fold_index is not None:
        if fold_index < 0 or K <= fold_index:
            raise IndexError('Invalid fold_index: {}'.format(fold_index))
        folds = [fold_index]
        print('Fold index {} specified'.format(fold_index))

    # train the model K times, one for each fold
    for i in folds:
        fold_key = _fold_key.format(depth_index, lr_index, i)

        # load model at previous state
        previous_depth_index = depth_index - 1
        if previous_depth_index < 0:
            fm.reload_model()
        else:
            fm.load_weights(_depth_key.format(previous_depth_index))
        fm.set_depth(depth_index)
        fm.compile_model(lr=lr)
        model = fm.get_model()

        print('[debug] batch: {}'.format(BATCH_SIZE))
        print('[debug] size: {}'.format(train_gens[i].n))
        print('[debug] steps: {}'.format(len(train_gens[i])))

        # train T epochs at a time
        start_epoch = 0
        save_interval = T
        while start_epoch < epochs:
            print('[debug] epoch {}'.format(start_epoch))
            target_epoch = start_epoch + save_interval
            if target_epoch > epochs:
                target_epoch = epochs
            result = model.fit_generator(
                train_gens[i],
                validation_data=val_gens[i],
                steps_per_epoch=len(train_gens[i]),
                validation_steps=len(val_gens[i]),
                workers=MULTIPROCESSING_WORKERS,
                use_multiprocessing=USE_MULTIPROCESSING,
                shuffle=True,
                epochs=target_epoch,
                initial_epoch=start_epoch,
            )
            start_epoch = target_epoch

            # update training history
            ch.append_history(result.history, fm.get_name(), fold_key)

            if SAVE_ALL_WEIGHTS:
                # save intermediate weights
                fm.save_weights(
                    _epoch_key.format(
                        depth_index,
                        lr_index,
                        i,
                        target_epoch,
                    ))

        # save final weights
        fm.save_weights(fold_key)

        # generate test results
        print('[debug] generating test results...')
        results.generate_test_result(fm,
                                     fold_key,
                                     load_weights=False,
                                     workers=MULTIPROCESSING_WORKERS,
                                     use_multiprocessing=USE_MULTIPROCESSING)

    if fold_index == None or fold_index == 4:
        # this will result in an error if fold_index == 4 and fold_index 0...3
        # have not been completed
        print('[debug] generating analysis of training process')
        for metric in analysis.metric_names.keys():
            analysis.analyze_lr(fm,
                                fm.get_name(),
                                depth_index,
                                lr_index,
                                lr,
                                metric,
                                exp=EXP)


def print_all_stats(train, test, folds):
    # Print stats for each train/test set
    def print_stats(collection):
        df = collection.df
        print('{:<3} patients / {:<4} images'.format(df.pid.unique().shape[0],
                                                     df.shape[0]))
        print(df.label.value_counts().to_string())

    print('Training/Validation Set'.center(80, '-'))
    print_stats(train)

    print('Test Set'.center(80, '-'))
    print_stats(test)

    print()
    print(
        'Note that OAP, OBS images in the training/validation set will be duplicated 5 times'
    )
    print('to solve the class imbalance issue')
    print()

    # Print number of images by fold by label (training data)
    stats = dict()
    for i, fold in enumerate(folds):
        counts = fold.df.label.value_counts()
        counts.loc['total'] = fold.df.shape[0]
        stats[i + 1] = counts
    stats = pd.DataFrame(stats)

    print('5-Fold Training Set Data'.center(80, '-'))
    print(stats.to_string(col_space=8))
    print()

    # Columnwise-print or cr_codes (training data)
    cr_codes_by_fold = list(sorted(fold.df.pid.unique()) for fold in folds)
    max_len = 0
    for codes in cr_codes_by_fold:
        if max_len < len(codes):
            max_len = len(codes)
    for i, _ in enumerate(folds):
        print('Fold {}'.format(i + 1).ljust(16), end='')
    print()
    print('-' * 80)
    for i in range(max_len):
        for codes in cr_codes_by_fold:
            if i < len(codes):
                print('{:<16d}'.format(codes[i]), end='')
            else:
                print('{:<16s}'.format(''), end='')
        print()
    print()


def run_by_lr(model_key,
              train_folds,
              test_collection,
              lr_index=None,
              fold_index=None):
    print(' MODEL: {} '.format(model_key).center(100, '#'))
    keras.backend.clear_session()
    models = FineModel.get_dict()
    fm = models[model_key]()
    train_gens, val_gens = fm.get_train_val_generators(train_folds)
    test_gen = fm.get_test_generator(test_collection)

    learning_rates = LEARNING_RATES
    if lr_index is not None:
        try:
            lr = LEARNING_RATES[lr_index]
            learning_rates = [lr]
        except IndexError as e:
            raise IndexError('Invalid lr_index: {}'.format(lr_index))
        print('Learning rate #{} ({}) specified'.format(lr_index, lr))

    for i, lr in enumerate(learning_rates):
        print('Starting training {} lr={}'.format(fm.get_name(),
                                                  lr).center(100, '-'))
        run_by_fold(fm, 0, i, EPOCHS, train_gens, val_gens, test_gen,
                    fold_index)


def main():
    train = cri.CrCollection.load().filter_by(
        dataset_index=0).tri_label().labeled()
    test = cri.CrCollection.load().filter_by(
        dataset_index=1).tri_label().labeled()
    if SAMPLE:
        train = train.sample(frac=0.1)
        test = test.sample(frac=0.1)
    folds = train.k_split(K, seed=K_SPLIT_SEED)

    print_all_stats(train, test, folds)

    for key in MODEL_KEYS:
        run_by_lr(key, folds, test, lr_index=LR_INDEX, fold_index=FOLD_INDEX)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        error += '\n'
        error += str(e)
        print(error)
        notify(error)
