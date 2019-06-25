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

    fmt = 'EXP{:02d}_D{:02d}_L{:02d}_F{:02d}_E{:03d}'
    parsed = parse.parse(fmt, key)

    if parsed:
        params['epochs'] = parsed[4]
    else:
        params['epochs'] = 100
        fmt = 'EXP{:02d}_D{:02d}_L{:02d}_F{:02d}'
        parsed = parse.parse(fmt, key)

    params['exp'] = parsed[0]
    params['depth'] = parsed[1]
    params['lr'] = parsed[2]
    params['fold'] = parsed[3]

    return params


def generate_test_result(fm: FineModel,
                         key,
                         save=True,
                         verbose=1,
                         load_weights=True):
    """
    Generates test results using all test images from db_index=1 (CAP TEST)

    If the weights are already loaded, set load_weight=False
    """
    params = parse_key(key)
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

    lrs = [0.0001, 0.00001]
    params['lr'] = lrs[params['lr']]

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
                                     description=description)

    if (verbose):
        print(key.center(80, '-'))
        print(result.describe())

    return result


def main():
    EXP_LIST = [1, 2]
    MODEL_KEY = 'mobileneta25'

    print('Generating results for existing weights (model={}) (exp={}) ...'.
          format(MODEL_KEY, EXP_LIST))

    fm = FineModel.get_dict()[MODEL_KEY]()
    fm.get_weight_keys()

    for key in fm.get_weight_keys():
        params = parse_key(key)
        if params['exp'] in EXP_LIST:
            generate_test_result(fm, key)


if __name__ == '__main__':
    main()
