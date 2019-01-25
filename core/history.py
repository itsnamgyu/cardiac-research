import os
import json
import sys

import pandas as pd

import core


DEFAULT_HISTORY_DIR = os.path.join(core.BASE_DIR, '.history')


def _get_history_dir(model_name, key, directory=None, makedirs=True):
    if directory:
        path = directory
    else:
        path = DEFAULT_HISTORY_DIR
    path = os.path.join(path, model_name)
    path = os.path.join(path, key + '.csv')

    os.makedirs(os.path.dirname(path), exist_ok=True)


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

    if old:
        df = pd.concat(old, new)
    else:
        df = new

    df.to_csv(path, index=False)


def load_history(model_name, key, directory=None):
    '''
    Return in pd.DataFrame
    '''
    path = _get_history_dir(model_name, key, directory)
    if os.path.exists(path):
        return pd.from_csv(path)
    else:
        return None


def get_average(histories):
    columns = histories[0].columns
    average = {}
    for column in columns:
        dfs = []
        for history in histories:
            dfs.append(history.loc[:, column])
        df = dfs.concat(dfs, axis=1)
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
