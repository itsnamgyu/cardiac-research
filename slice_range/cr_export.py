import os
import json
import shutil
from typing import Dict
import re
import glob
import argparse
import itertools

import numpy as np
import scipy.ndimage
import progress.bar
import imageio
import matplotlib.pyplot as plt
import pandas as pd

import cr_interface as cri


DATA_DIR = os.path.join(cri.PROJECT_DIR, 'data/data')
TRAIN_CSV = os.path.join(cri.PROJECT_DIR, 'analysis/data_spec_train.csv')
VALIDATION_CSV = os.path.join(cri.PROJECT_DIR, 'analysis/data_spec_validation.csv')
TEST_CSV = os.path.join(cri.PROJECT_DIR, 'analysis/data_spec_test.csv')


def cut_list(values, ratio):
    '''
    Get a tuple of two lists that each contain (ratio, 1 - ratio) of the original list
    '''
    sorted_list = list(sorted(set(values)))
    n1 = int(len(sorted_list) * ratio)
    # n2 = len(sorted_list) - n1  # unused
    first = sorted_list[:n1]
    second = sorted_list[n1:]

    return first, second


def split_patients(datasets=[0], 
                   train_datasets=None,
                   test_datasets=None,
                   test_split=0.2,
                   validation_split=0.2):
    '''
    Splits pids into train, test, validation according to split details

    # Arguments
    - datasets: list of dataset indices to use for train + eval
    - test_split: ratio of test patients over total
    - validation_split: ratio validation patients over training
    - train_datasets: list of dataset indices to use for training
    - test_datasets: list of dataset indices to use for testing

    # Returns
    {
        'train': [ (db_index, pid) ... ],
        'validaiton': ...,
        'test': ...,
    }
    '''
    metadata = cri.load_metadata()

    pids = dict(
        train=[],
        validation=[],
        test=[])

    if train_datasets is None and test_datasets is None:
        # Split `datasets`
        all_pids = []
        for cr_code in metadata:
            cr = cri.parse_cr_code(cr_code)
            if cr[0] in datasets:
                all_pids.append((cr[0], cr[1]))
        pids['test'], train_pids = cut_list(all_pids, test_split)
        pids['validation'], pids['train'] = cut_list(train_pids, test_split)
    elif train_datasets is None or test_datasets is None:
        raise Exception('train_datasets and test_datasets must be passed in conjuction')
    else:
        # Use `test_datasets`, split `train_datasets`
        train_pids = []
        for cr_code in metadata:
            cr = cri.parse_cr_code(cr_code)
            if cr[0] in train_datasets:
                train_pids.append((cr[0], cr[1]))
            if cr[0] in test_datasets:
                test_pids.append((cr[0], cr[1]))
        pids['test'] = sorted(list(set(test_pids)))
        pids['validation'], pids['train'] = cut_list(train_pids, validation_split)

    return pids


def export_images(tri_label=False,
                  datasets=[0], 
                  train_datasets=None,
                  test_datasets=None,
                  test_split=0.2,
                  validation_split=0.2,
                  export_dir=DATA_DIR):
    '''
    Exports images for training and evaluation. Splits data by patient and
    places them in `export_dir` according to their cooresponding datasets
    (test/train/val) and labels.

    # Arguments
    - datasets: list of dataset indices to use for train + eval
    - test_split: ratio of test patients over total
    - validation_split: ratio validation patients over training
    - train_datasets: list of dataset indices to use for training
    - test_datasets: list of dataset indices to use for testing

    Note that train/test_datasets overrides datasets and train_split
    '''
    print('Exporting images...')

    metadata = cri.load_metadata()
    pids = split_patients(datasets, train_datasets, test_datasets,
                          test_split, validation_split)

    if datasets == None:
        all_datasets = train_datasets + test_dataset
    else:
        all_datasets = datasets
        
    print('Number of Patients')
    for key in pids:
        print('{:<10s}: {}'.format(key, len(pids[key])))

    if tri_label:
        labels = ['OAP', 'IN', 'OBS']
    else:
        labels = ['OAP', 'AP', 'MD', 'BS', 'OBS']
    
    # init dataframe for image statistics
    subcolumns = labels + ['TOTAL']
    index = ('Image Count', 'Percentages', 'Images Per Patient')

    train_columns = pd.MultiIndex.from_product([['Original', 'Augmented'], subcolumns])
    train_df = pd.DataFrame(np.zeros((3, 8)), columns=train_columns, index=index)

    test_columns = pd.Index(subcolumns)
    test_df = pd.DataFrame(np.zeros((3, 4)), columns=test_columns, index = index)

    # count total # of cr_codes to consider (for loading bar)
    count = len([1 for cr_code in metadata if cri.parse_cr_code(cr_code)[0] in all_datasets])
    bar = progress.bar.IncrementalBar('Copying Images...', max=count)

    # check IMAGES_DIR
    if glob.glob(os.path.join(export_dir, '**/*.jpg'), recursive=True):
        print('WARNING: there are existing images in "{}"'.format(export_dir))

    # copy images
    unlabeled = []
    done = []
    for cr_code, info in metadata.items():
        cr = cri.parse_cr_code(cr_code)
        if cr[0] not in all_datasets:
            continue
        bar.next()
        done.append(cr_code)

        try:
            label = info['label']
        except KeyError:
            unlabeled.append(cr_code)
            label = 'nan'

        if tri_label and label in ['ap', 'md', 'bs']:
            label = 'in'

        # get dataset
        for key in pids:
            if (cr[0], cr[1]) in pids[key]:
                dataset = key
                break
        else:
            raise Exception('invalid implementation')

        if key != 'test' and label == 'nan':
            raise Exception('Non testset image {} unlabeled'.format(cr_code))

        dirname = export_dir
        dirname = os.path.join(dirname, dataset)
        dirname = os.path.join(dirname, label)
        os.makedirs(dirname, exist_ok=True)

        # hotfix - oversample oap, obs
        n_oversample = 5

        dests = []
        if tri_label and label in ['oap', 'obs'] and dataset != 'test':
            for i in range(n_oversample):
                dests.append(os.path.join(dirname, '{}_{:02d}.jpg'.format(cr_code, i)))
        else:
            dests.append(os.path.join(dirname, '{}.jpg'.format(cr_code)))
        src = os.path.join(cri.DATABASE_DIR, cr_code + '.jpg')

        train_df[('Original', label.upper())] += 1
        train_df[('Augmented', label.upper())] += len(dests)

        for dest in dests:
            shutil.copy(src, dest)

    bar.finish()

    # dataset summary overview
    total = train_df.loc['Image Count', 'Original'].sum()
    for i in range(3):
        train_df.loc['Percentages', train_columns[i]] = \
                train_df.loc['Image Count', train_columns[i]] / total * 100
    total = train_df.loc['Image Count', 'Augmented'].sum()
    for i in range(4, 7):
        train_df.loc['Percentages', train_columns[i]] = \
                train_df.loc['Image Count', train_columns[i]] / total * 100
    train_df.loc['Images Per Patient'] = train_df.loc['Image Count'] / len(pids['train'])
    for label in ['Original', 'Augmented']:
        train_df.loc[:,(label, 'TOTAL')] = train_df.loc[:, label].sum(axis=1)

    total = test_df.loc['Image Count'].sum()
    test_df.loc['Percentages'] = test_df.loc['Image Count'] / total * 100
    test_df.loc['Images Per Patient'] = test_df.loc['Image Count'] / len(pids['test'])
    test_df['TOTAL'] = test_df.sum(axis=1)

    # save dataset info to csvs
    train_csv = os.path.join(os.path.dirname(spec_csv), 'train_' + os.path.basename(spec_csv))
    test_csv = os.path.join(os.path.dirname(spec_csv), 'test_' + os.path.basename(spec_csv))
    train_df.to_csv(train_csv)
    test_df.to_csv(test_csv)
    print('saved train dataset info to' + train_csv)
    print('saved test dataset info to' + test_csv)

    # print copied images
    print('Copied')
    print(done)
    print('Unlabeled')
    print(unlabeled)


def main():
    export_images(tri_label=True)


if __name__ == "__main__":
    main()
