import os
import json
import sys
import traceback
from collections import defaultdict

import pandas as pd

import core


DEFAULT_HISTORY_DIR = os.path.join(core.BASE_DIR, '.history')


def _get_history_module_dir(directory=None):
    if directory:
        return directory
    else:
        return DEFAULT_HISTORY_DIR


def _get_history_dir(model_name, key, directory=None, makedirs=True):
    path = _get_history_module_dir(directory)
    path = os.path.join(path, model_name)
    path = os.path.join(path, key + '.csv')

    os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


def save_history(history, model_name, key, directory=None):
    '''
    history: keras_model.fit().history
        contains val_loss, val_acc, loss, acc
    '''
    path = _get_history_dir(model_name, key, directory)
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)


def append_history(history, model_name, key, directory=None):
    '''
    Creates history if it doesn't exist

    history: keras_model.fit().history
        contains val_loss, val_acc, loss, acc
    '''
    old = load_history(model_name, key, directory)

    path = _get_history_dir(model_name, key, directory)
    new = pd.DataFrame(history)

    if old is not None:
        df = pd.concat([old, new])
    else:
        df = new

    df.to_csv(path, index=False)


def load_history(model_name, key, directory=None):
    '''
    Return in pd.DataFrame
    '''
    path = _get_history_dir(model_name, key, directory)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


def get_keys(directory=None):
    path = _get_history_module_dir(directory)
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
    


def reset_history(model_name, key, directory=None):
    '''
    Delete history file
    '''
    path = _get_history_dir(model_name, key, directory)
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
