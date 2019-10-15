import os
import json
import sys
import traceback
from collections import defaultdict
import warnings
import shutil

import pandas as pd

import core
from . import utils

HISTORY_DIR = 'cr_train_val_history'


def _get_history_dir(exp_dir=None):
    if exp_dir is None:
        exp_dir = '.'

    utils.validate_exp_dir(exp_dir)

    return os.path.abspath(os.path.join(exp_dir, HISTORY_DIR))


def _get_history_path(model_name, key, exp_dir=None, makedirs=True):
    path = _get_history_dir(exp_dir)
    path = os.path.join(path, model_name)
    path = os.path.join(path, key + '.csv')

    os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


def save_history(history, model_name, key, exp_dir=None):
    '''
    history: keras_model.fit().history
        contains val_loss, val_acc, loss, acc
    '''
    path = _get_history_path(model_name, key, exp_dir)
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)


def append_history(history, model_name, key, exp_dir=None):
    """Append history to existing file, mainly for use when you train N epochs
    at a time, calling model.fit() multiple times.

    Arguments:
    - history: keras_model.fit().history (should contain val_loss, val_acc,
      loss, acc)
    """
    old = load_history(model_name, key, exp_dir)

    path = _get_history_path(model_name, key, exp_dir)
    new = pd.DataFrame(history)

    if old is not None:
        df = pd.concat([old, new])
    else:
        df = new

    df.to_csv(path, index=False)


def reset_history(model_name, key, exp_dir=None):
    """Reset history, mainly for use before append_history to prevent
    appending to an existing, complete history file.
    """
    path = _get_history_path(model_name, key, exp_dir)
    utils.remove_safe(path)


def load_history(model_name, key, exp_dir=None):
    '''Load history as a DataFrame
    '''
    path = _get_history_path(model_name, key, exp_dir)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


def get_keys(exp_dir=None):
    path = _get_history_dir(exp_dir)
    model_dirs = os.listdir(path)
    d = defaultdict(list)
    for model_dir in model_dirs:
        model_name = model_dir
        model_path = os.path.join(path, model_dir)
        if not os.path.isdir(model_path):
            continue
        filenames = os.listdir(model_path)
        for filename in filenames:
            file_path = os.path.join(model_path, filename)
            key, ext = os.path.splitext(filename)
            if ext == '.csv':
                d[model_name].append(key)

    for keys in d.values():
        keys.sort()

    return dict(d)


def reset_history(model_name, key, exp_dir=None):
    '''
    Delete history file
    '''
    path = _get_history_path(model_name, key, exp_dir)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            traceback.print_exc()
            warnings.warn('error: could not delete {}:\n{}'.format(path, e))


def get_average(histories):
    columns = histories[0].columns
    average = {}
    for column in columns:
        dfs = []
        for history in histories:
            dfs.append(history.loc[:, column])
        df = pd.concat(dfs, axis=1)
        average[column] = df.mean(axis=1)
    return pd.DataFrame(average)


def get_early_stop_index_and_value(history, metric='val_loss', patience=10):
    '''
    Extremely unoptimized: O(n) for n=epochs
    '''
    series = history.loc[:, metric]
    min_value = sys.maxsize
    min_index = -1

    for i, value in enumerate(series):
        if value < min_value:
            min_value = value
            min_index = i
        if min_index + patience < i:
            return min_index, min_value

    return None, None
