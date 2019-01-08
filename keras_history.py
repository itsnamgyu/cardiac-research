import os
import argparse
import tempfile
import warnings
import shutil
import argparse
from collections import defaultdict
import json

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import keras
from tqdm import tqdm
import requests
import pandas as pd

import keras_utils as ku
import cr_interface as cri
import lib


HISTORY_DIR = os.path.join(cri.PROJECT_DIR, 'history')
TEMP_DIR = os.path.join(cri.PROJECT_DIR, '.keras_history_temp')
TEST_RESULT_DIR = os.path.join(HISTORY_DIR, 'test_result')


def _get_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def _load_test_results():
    if os.path.exists(TEST_RESULTS):
        with open(TEST_RESULTS) as f:
            return json.load(f)
    else:
        return dict()


def _save_test_results(results):
    with open(TEST_RESULTS) as f:
        json.save(results, f)


def get_history_dir(model, mkdir=True):
    subdir = '{}_{}'.format(model.name.replace('.', '_'), len(model.layers))
    path = os.path.join(HISTORY_DIR, subdir)
    if mkdir:
        os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def get_history_path(model, lr_index, epochs, split_index=None):
    dirname = get_history_dir(model)
    if split_index is not None:
        basename = 'L{:02d}_E{:04d}_S{:02d}.json'.format(lr_index, epochs, split_index)
    else:
        basename = 'L{:02d}_E{:04d}.json'.format(lr_index, epochs)
    return os.path.join(dirname, basename)


def save_history(history, model, lr_index, epochs, split_index=None):
    with open(get_history_path(model, lr_index, epochs, split_index), 'w') as f:
        json.dump(history, f)


def load_history(model, lr_index, epochs, split_index=None):
    with open(get_history_path(model, lr_index, epochs, split_index)) as f:
        return json.load(f)


def load_average_history(model, lr_index, epochs, k=5):
    histories = []
    for i in range(k):
        histories.append(pd.DataFrame(load_history(model, lr_index, epochs, i)))
    return pd.concat(histories).groupby(level=0).mean()


def get_test_result_path(model, mkdir=True):
    if mkdir:
        os.makedirs(TEST_RESULT_DIR, exist_ok=True)
    basename = '{}_{}.json'.format(model.name.replace('.', '_'), len(model.layers))
    return os.path.join(TEST_RESULT_DIR, basename)


def load_test_result(model, mkdir=True):
    with open(get_test_result_path(model)) as f:
        return json.load(f)


def save_test_result(model, loss, accuracy, lr, epochs, mkdir=True):
    result = dict(
        loss=str(loss),
        accuracy=str(accuracy),
        lr=str(lr),
        epochs=str(epochs),
    )
    with open(get_test_result_path(model), 'w') as f:
        json.dump(result, f)
