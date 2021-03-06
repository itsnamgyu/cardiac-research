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
import cr_analysis as cra

import core.fine_model
from core.fine_model import FineModel


def parse_key(key):
    params = {}

    fmt = 'EXP{:d}_D{:d}_L{:d}_F{:d}_E{:d}'  # hotfix for more digits
    parsed = parse.parse(fmt, key)

    if parsed:
        params['epochs'] = parsed[4]
    else:
        params['epochs'] = 100
        fmt = 'EXP{:d}_D{:d}_L{:d}_F{:d}'  # hotfix for more digits
        parsed = parse.parse(fmt, key)

    params['exp'] = parsed[0]
    params['depth'] = parsed[1]
    params['lr'] = parsed[2]
    params['fold'] = parsed[3]

    return params


def generate_test_result(fm: FineModel,
                         key,
                         lr_list,
                         save=True,
                         verbose=1,
                         workers=4,
                         use_multiprocessing=False,
                         load_weights=True,
                         total_epochs=None):
    """
    Generates test results using all test images from db_index=1 (CAP TEST)

    If the weights are already loaded, set load_weight=False
    """
    params = parse_key(key)
    if 'epochs' not in params:
        params['epochs'] = total_epochs
    description = ''

    exp = params['exp']
    del params['exp']
    if exp == 1:
        description = 'Failed Experiment'
    if exp == 2:
        description = 'Original (seed=1)'
    if exp == 3:
        description = 'Original (seed=2)'
    if exp == 4:
        description = 'Failed Experiment'
    if exp == 5:
        description = 'No Crop (seed=1)'

    lr_index = params['lr']
    params['lr'] = lr_list[lr_index]

    dt = datetime.datetime.now()
    description += '\n'
    description += 'Analyzed on: {}'.format(dt.strftime("%Y-%m-%d %H:%M:%S"))

    fm.load_weights(key)
    fm.compile_model()
    test = cri.CrCollection.load().filter_by(
        dataset_index=1).tri_label().labeled()
    result = fm.generate_test_result(test,
                                     verbose=verbose,
                                     save_to_key=key,
                                     params=params,
                                     workers=workers,
                                     use_multiprocessing=use_multiprocessing,
                                     description=description)

    if (verbose):
        print(key.center(80, '-'))
        print(result.describe())

    return result


def main():
    # set LR_LIST based on the specifications of the experiment
    EXP_LIST = [1, 2]
    LR_LIST = [0.0001, 0.00001]
    MODEL_KEY = 'mobileneta25'

    print('Generating results for existing weights (model={}) (exp={}) ...'.
          format(MODEL_KEY, EXP_LIST))

    fm = FineModel.get_dict()[MODEL_KEY]()
    fm.get_weight_keys()

    for key in fm.get_weight_keys():
        params = parse_key(key)
        if params['exp'] in EXP_LIST:
            generate_test_result(fm, key, lr_list=LR_LIST)


if __name__ == '__main__':
    main()
