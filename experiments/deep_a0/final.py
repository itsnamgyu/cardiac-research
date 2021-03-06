"""
Module to train final model based on manually selected learning
rate and epoch.

Only supports depth=0 (top-layer tuning)
"""
#!/usr/bin/env python
# coding: utf-8

import matplotlib as mpl
# Don't display mpl windows (will cause error in non-gui environment)
mpl.use('Agg')

import traceback

import core.history as ch
from core.fine_model import FineModel

import cr_interface as cri

from lib import notify
import results
import stats

from params_final import (BATCH_SIZE, SAVE_ALL_WEIGHTS, T, BALANCE,
                          DEPTH_INDEX, LEARNING_RATE, EPOCHS, SAMPLE, MODEL_KEY,
                          USE_MULTIPROCESSING, MULTIPROCESSING_WORKERS)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def run(fm: FineModel,
        training_set: cri.CrCollection,
        epochs=EPOCHS,
        depth_index=DEPTH_INDEX,
        batch_size=BATCH_SIZE,
        augment_factor=BALANCE,
        learning_rate=LEARNING_RATE,
        save_interval=T,
        use_multiprocessing=USE_MULTIPROCESSING,
        workers=MULTIPROCESSING_WORKERS):
    """
    Train model and evalute results. Output files are saved to
    `output/<model_key>/D00_FINAL/`. These include:

    - Intemediate model weights
    - Final model weights
    - Test set result
    - Training history
    """
    _depth_key = 'D{:02d}_FINAL'
    instance_key = _depth_key.format(depth_index)
    _epoch_key = instance_key + "_E{:03}"

    if depth_index >= 1:
        fm.load_weights(_depth_key.format(depth_index - 1))
    fm.set_depth(depth_index)
    fm.compile_model(lr=learning_rate)
    model = fm.get_model()

    gen = fm.get_directory_iterator(training_set,
                                    'train',
                                    augment=True,
                                    augment_factor=augment_factor,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    verbose=1,
                                    title='final training set')

    print("[DEBUG] Batch size: {}".format(batch_size))
    print("[DEBUG] Number of images: {}".format(gen.n))
    print("[DEBUG] Steps: {}".format(len(gen)))

    # Train T epochs at a time
    start_epoch = 0
    # Reset training history
    ch.reset_history(fm.get_key(), instance_key)
    while start_epoch < epochs:
        print("[DEBUG] Starting epoch {}".format(start_epoch))
        target_epoch = start_epoch + save_interval
        if target_epoch > epochs:
            target_epoch = epochs
        result = model.fit_generator(
            gen,
            steps_per_epoch=len(gen),
            shuffle=True,
            epochs=target_epoch,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            initial_epoch=start_epoch,
        )
        start_epoch = target_epoch

        # Update training history every T epochs
        ch.append_history(result.history, fm.get_key(), instance_key)

        # Save intermediate weights every T epochs
        if SAVE_ALL_WEIGHTS:
            epoch_key = _epoch_key.format(target_epoch)
            fm.save_weights(epoch_key)

    # Save final weights
    fm.save_weights(instance_key)

    # Generate test results
    print("[DEBUG] Generating test results...")
    results.generate_test_result(
        fm,
        instance_key,
        learning_rate,
        epochs,
        load_weights=False,
        workers=MULTIPROCESSING_WORKERS,
        use_multiprocessing=USE_MULTIPROCESSING,
    )


def load_training_set():
    train = cri.CrCollection.load().filter_by(
        dataset_index=0).tri_label().labeled()

    if SAMPLE and SAMPLE != 1:
        print("Sampling {:.0%} of original training dataset".format(SAMPLE))
        train = train.sample(frac=SAMPLE)

    stats.print_collection_stats(train, "Training set")

    return train


def main():
    fm = FineModel.get_dict()[MODEL_KEY]()
    training_set = load_training_set()
    run(fm, training_set)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error = traceback.format_exc()
        error += "\n"
        error += str(e)
        print(error)
        notify(error)
