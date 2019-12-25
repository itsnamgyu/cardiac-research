'''
This module assumes the default parameters used in deep_3 experiments
'''

import core
import imp
import os
import shutil
import parse
import datetime

import cr_interface as cri

import core.fine_model
from core.fine_model import FineModel

from params import *


def parse_key(key):
    params = {}

    fmt = 'D{:d}_L{:d}_F{:d}_E{:d}'
    parsed = parse.parse(fmt, key)

    if parsed:
        params['epochs'] = parsed[3]
    else:
        fmt = 'D{:d}_L{:d}_F{:d}'
        parsed = parse.parse(fmt, key)

    params['depth'] = parsed[0]
    params['lr'] = parsed[1]
    params['fold'] = parsed[2]

    return params


def generate_test_result(fm: FineModel,
                         key: str,
                         lr: float,
                         epochs: int,
                         save=True,
                         verbose=1,
                         workers=4,
                         params=None,
                         use_multiprocessing=False,
                         load_weights=True,
                         description=''):
    """
    Generates test results using all test images from db_index=1 (CAP TEST)

    If the weights are already loaded, set load_weight=False
    """
    if params is None:
        params = dict()
    params['lr'] = lr
    params['epochs'] = epochs

    description = str(description)
    description += '\n'
    description += 'Analyzed on: {}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if load_weights:
        fm.load_weights(key)
    fm.compile_model()
    test = cri.CrCollection.load().filter_by(
        dataset_index=1).tri_label().labeled()

    save_to_instance_key = key if save else None
    result = fm.generate_test_result(test,
                                     verbose=verbose,
                                     save_to_instance_key=save_to_instance_key,
                                     params=params,
                                     workers=workers,
                                     use_multiprocessing=use_multiprocessing,
                                     description=description)
    if verbose:
        print(key.center(80, '-'))
        print(result.describe())

    return result
