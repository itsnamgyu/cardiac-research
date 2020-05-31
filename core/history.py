import os
import sys

import pandas as pd

from . import paths
from . import utils


def _get_history_path(model_key, instance_key, exp_key=None, makedirs=True):
    path = paths.get_history_path(exp_key, model_key, instance_key)
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def fix_history_keys(history):
    """Undo changes to history keys in Keras 2.3.0 (acc -> accuracy)
    """
    replacements = {
        "val_accuracy": "val_acc",
        "accuracy": "acc",
    }
    for before, after in replacements.items():
        if before in history:
            history[after] = history[before]
            del history[before]


def save_history(history, model_key, instance_key, exp_key=None):
    """
    history: keras_model.fit().history (contains val_loss, val_acc, loss, acc)
    """
    path = _get_history_path(model_key,
                             instance_key,
                             exp_key=exp_key,
                             makedirs=True)
    fix_history_keys(history)
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)


def append_history(history, model_key, instance_key, exp_key=None):
    """Append history to existing file, mainly for use when you train N epochs
    at a time, calling model.fit() multiple times.

    Arguments:
    - history: keras_model.fit().history (should contain val_loss, val_acc,
      loss, acc)
    """
    old = load_history(model_key, instance_key, exp_key=exp_key)
    fix_history_keys(history)
    new = pd.DataFrame(history)
    if old is not None:
        df = pd.concat([old, new])
    else:
        df = new
    save_history(df, model_key, instance_key, exp_key=exp_key)


def reset_history(model_key, instance_key, exp_key=None):
    """Reset history, mainly for use before append_history to prevent
    appending to an existing, complete history file.
    """
    path = _get_history_path(model_key, instance_key, exp_key=exp_key)
    utils.remove_safe(path)


def load_history(model_key, instance_key, exp_key=None):
    """Load history as a DataFrame
    """
    path = _get_history_path(model_key, instance_key, exp_key=exp_key)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


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
    """Extremely unoptimized: O(n) for n=epochs
    """
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
