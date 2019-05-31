# IN PROGRESS!!!

import os
import sys

import pandas as pd
import cr_interface as cri

import core


DEFAULT_RESULT_DIR = os.path.join(core.BASE_DIR, '.result')
DEFAULT_TEMP_TEST_IMAGE_DIR = os.path.join(
    DEFAULT_RESULT_DIR, 'temp_test_images')


class Result:
    @staticmethod
    def generate(model, test_collection: cri.CrCollection):
        """
        Evaluate and generate results for the model and the given test data

        :param model:
        Keras model to use to generate predictions

        :param test_collection:

        CrCollection containing the test data
        :return:
        """
        test_collection.export(dest=DEFAULT_TEMP_TEST_IMAGE_DIR)

    @staticmethod
    def _get_result_dir(model_name, key, directory=None, makedirs=True):
        if directory:
            path = directory
        else:
            path = DEFAULT_RESULT_DIR
        path = os.path.join(path, model_name)
        path = os.path.join(path, key + '.json')

        os.makedirs(os.path.dirname(path), exist_ok=True)

        return path

    def save(self, model_name, key, directory=None):
        """
        :param result:
        :param model_name:
        :param key:
        :param directory:
        :return:
        """
        path = Result._get_result_dir(model_name, key, directory)
        df = pd.DataFrame(result)
        df.to_csv(path, index=False)


def append_result(result, model_name, key, directory=None):
    '''
    Creates result if it doesn't exist

    result: keras_model.fit().result
        contains val_loss, val_acc, loss, acc
    '''
    old = load_result(model_name, key, directory)

    path = _get_result_dir(model_name, key, directory)
    new = pd.DataFrame(result)

    if old is not None:
        df = pd.concat([old, new])
    else:
        df = new

    df.to_csv(path, index=False)


def load_result(model_name, key, directory=None):
    '''
    Return in pd.DataFrame
    '''
    path = _get_result_dir(model_name, key, directory)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None


def get_average(histories):
    columns = histories[0].columns
    average = {}
    for column in columns:
        dfs = []
        for result in histories:
            dfs.append(result.loc[:, column])
        df = pd.concat(dfs, axis=1)
        average[column] = df.mean(axis=1)
    return pd.DataFrame(average)


def get_early_stop_index_and_value(result, metric='val_loss', patience=10):
    '''
    Extremely unoptimized: O(n) for n=epochs
    '''
    series = result.loc[:, metric]
    min_value = sys.maxsize
    min_index = -1

    for i, value in enumerate(series):
        if value < min_value:
            min_value = value
            min_index = i
        if min_index + patience < i:
            return min_index, min_value

    return None, None
