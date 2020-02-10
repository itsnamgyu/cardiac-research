#!/usr/bin/env python
# coding: utf-8
import matplotlib as mpl

# Don't display mpl windows (will cause error in non-gui environment)
mpl.use("Agg")

import keras
import traceback
import argparse
import time
import datetime
import os

import core.history as ch
from core.fine_model import FineModel

import cr_interface as cri

from lib import notify
import stats

from params import (BATCH_SIZE, K, SAVE_ALL_WEIGHTS, T, BALANCE,
                    LEARNING_RATES, EPOCHS, SAMPLE, K_SPLIT_SEED, MODEL_KEYS,
                    FOLD_INDEX, USE_MULTIPROCESSING, MULTIPROCESSING_WORKERS)

RECORD_FILE = "time.csv"


def run_all_folds(
        fm: FineModel,
        depth_index,
        lr_index,
        epochs,
        train_gens,
        val_gens,
        fold_index=None,
        generate_plots=True,
):
    """
    Train the model (frozen at some depth) for all five folds OR a specific
    fold. Weights, history and results are saved using instance keys in the
    following format:

    D01_L03_F01:
    1st freeze depth, 3rd learning rate, fold 1
    D01_L03_F01_E025:
    1st freeze depth, 3rd learning rate, fold 1, trained until the 25th epoch

    :param fm:
    FineModel to train, i.e., the base network to train on

    :param depth_index:
    The INDEX of the "freeze depth" for the given FineModel

    :param lr_index:
    The INDEX of the learning rate, i.e., lr = LEARNING_RATES[lr_index]

    :param epochs:
    Number of epochs to train. MUST BE MULTIPLE OF 5.

    :param train_gens:
    List of train ImageDataGenerators for each fold

    :param val_gens:
    List of validation ImageDataGenerators for each fold

    :param fold_index
    If specified, will only run the specific fold index
    """
    _depth_key = "D{:02}"
    _fold_key = _depth_key + "_L{:02}_F{:02}"
    _epoch_key = _fold_key + "_E{:03}"

    lr = LEARNING_RATES[lr_index]

    folds = range(K)
    if fold_index is not None:
        if fold_index < 0 or K <= fold_index:
            raise IndexError("Invalid fold_index: {}".format(fold_index))
        folds = [fold_index]
        print("Fold index {} specified".format(fold_index))

    # Train model K times, one for each fold
    for i in folds:
        fold_key = _fold_key.format(depth_index, lr_index, i)

        # Load model at previous state
        previous_depth_index = depth_index - 1
        if previous_depth_index < 0:
            fm.reload_model()
        else:
            fm.load_weights(_depth_key.format(previous_depth_index))
        fm.set_depth(depth_index)
        fm.compile_model(lr=lr)
        model = fm.get_model()

        print("[DEBUG] Batch size: {}".format(BATCH_SIZE))
        print("[DEBUG] Number of images: {}".format(train_gens[i].n))
        print("[DEBUG] Steps: {}".format(len(train_gens[i])))

        # Train T epochs at a time
        start_epoch = 0
        save_interval = T
        # Reset training history
        ch.reset_history(fm.get_key(), fold_key)
        while start_epoch < epochs:
            print("[DEBUG] Starting epoch {}".format(start_epoch))
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

            # Update training history every T epochs
            ch.append_history(result.history, fm.get_key(), fold_key)

            # Save intermediate weights every T epochs
            if SAVE_ALL_WEIGHTS:
                epoch_key = _epoch_key.format(depth_index, lr_index, i,
                                              target_epoch)
                fm.save_weights(epoch_key)

        # Save final weights
        fm.save_weights(fold_key)


def run_all_lrs(model_key,
                train_folds,
                depth_index=0,
                lr_index=None,
                fold_index=None,
                reverse=False):
    print(" MODEL: {} ".format(model_key).center(100, "#"))
    keras.backend.clear_session()
    models = FineModel.get_dict()
    fm: FineModel = models[model_key]()
    train_gens, val_gens = fm.get_train_val_generators(train_folds,
                                                       augment_factor=BALANCE)

    enumerated_learning_rates = list(enumerate(LEARNING_RATES))
    if lr_index is not None:
        try:
            elr = enumerated_learning_rates[lr_index]
            enumerated_learning_rates = [elr]
        except IndexError as e:
            raise IndexError("Invalid lr_index: {}".format(lr_index))
        print("Learning rate #{} ({}) specified".format(elr[0], elr[1]))

    if reverse:
        enumerated_learning_rates.reverse()

    for lr_index, lr in enumerated_learning_rates:
        print("Starting training {} lr={}".format(fm.get_key(),
                                                  lr).center(100, "-"))
        print("Skip!")  # debug
        start = time.time()
        """
        run_all_folds(fm,
                      depth_index,
                      lr_index,
                      EPOCHS,
                      train_gens,
                      val_gens,
                      fold_index,
                      generate_plots=False)
        """
        time.sleep(5)
        end = time.time()
        if not os.path.exists(RECORD_FILE):
            with open(RECORD_FILE, "w") as f:
                f.write("Learning Rate,Experiment Date,Duration,Mode\n")

        mode = 'Normal'
        if reverse:
            mode = 'Reverse'
        elif lr_index is not None:
            mode = 'Individual'

        with open(RECORD_FILE, "a") as f:
            now = datetime.datetime.now()
            fmt = "{lr},{now},{duration:04.4f},{mode}\n"
            line = fmt.format(lr=lr,
                              now=str(now),
                              duration=end - start,
                              mode=mode)
            f.write(line)


def main():
    # Accept learning rate as argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--lr',
                        metavar='N',
                        help="Run single learning rate with specific index",
                        type=int,
                        default=None)
    parser.add_argument('-r',
                        '--reverse',
                        help="Run all learning rates in reverse",
                        action='store_true')
    args = parser.parse_args()
    reverse = args.reverse
    lr_index = args.lr

    if lr_index is not None:
        # Check learning rate index is specified
        if not 0 <= lr_index < len(LEARNING_RATES):
            print('Invalid learning rate index')
            return

    # Load datasets and print stats
    train = cri.CrCollection.load().filter_by(
        dataset_index=0).tri_label().labeled()
    if SAMPLE and SAMPLE != 1:
        print("Sampling {:.0%} of original training dataset".format(SAMPLE))
        train = train.sample(frac=SAMPLE)
    folds = train.k_split(K, seed=K_SPLIT_SEED)

    stats.print_collection_stats(train, "Training/Validation Set")
    stats.print_fold_stats(folds)

    for key in MODEL_KEYS:
        run_all_lrs(key,
                    folds,
                    lr_index=lr_index,
                    reverse=reverse,
                    fold_index=FOLD_INDEX)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        error += "\n"
        error += str(e)
        print(error)
        notify(error)
