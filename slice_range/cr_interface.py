import os
import json
import shutil
from typing import Dict
import re
import glob
import argparse

import numpy as np
import scipy.ndimage
import progress.bar
import imageio
import matplotlib.pyplot as plt
import pandas as pd

module_dir = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(module_dir, 'data/database')
METADATA_FILE = os.path.join(module_dir, 'data/metadata.json')
IMAGES_DIR = os.path.join(module_dir, 'data/data')
RESULTS_DIR = os.path.join(module_dir, 'results')
SPEC_CSV = os.path.join(module_dir, 'analysis/images_spec.csv')


'''
cr_metadata.json
{
    'D00_P00000101_P00_S00':
    {
            'original_filepath': 'cap_challenge/DET0000101/DET0000101_SA12_ph0.dcm',
            'original_name': 'DET0000101_SA12_ph0',
            'label': 'obs'
    }
    ...
}
'''


def load_metadata() -> Dict[str, Dict[str, str]]:
    '''
    Load or initialize metadata file
    '''

    if os.path.isfile(METADATA_FILE):
        try:
            with open(METADATA_FILE) as f:
                metadata: Dict[str: Dict[str: str]] = json.load(f)
        except json.JSONDecodeError:
            raise Exception('corrupt metadata file: {}'.format(METADATA_FILE))
    else:
        print('no metadata file')
        print('initializing new metadata')
        metadata: Dict[str: Dict[str: str]] = {}

    return metadata


def save_metadata(metadata: Dict[str, Dict[str, str]]) -> None:
    if os.path.isfile(METADATA_FILE):
        shutil.copyfile(METADATA_FILE, '.' + METADATA_FILE + '.bak')

    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

    print('Metadata file {} has been updated (including labels etc.)'.format(
        METADATA_FILE))


def get_cr_code(dataset_index, patient_index, phase_index, slice_index):
    cr_code = 'D%02d_P%08d_P%02d_S%02d' % (dataset_index, patient_index,
                                           phase_index, slice_index)
    return cr_code


re_cr_code = re.compile('D([0-9]{2})_P([0-9]{8})_P([0-9]{2})_S([0-9]{2})')


def extract_cr_code(string):
    '''
    Return cr_code from string that contains one
    '''
    return re_cr_code.search(string).group(0)


def parse_cr_code(cr_code, match=True):
    '''
    Return: (dataset_index, patient_index, phase_index, slice_index)
    '''
    if match:
        match = re_cr_code.match(cr_code)
    else:
        match = re_cr_code.search(cr_code)

    if not match:
        raise Exception('could not parse cr code {}'.format(cr_code))

    return tuple(map(lambda index: int(index), match.groups()))


def prepare_images(tri_label=False, n_rotation=6, n_oversample=1,
                   datasets=[0], train_split=0.8,
                   train_datasets=None, test_datasets=None,
                   spec_csv=SPEC_CSV):
    '''
    Arguments
    n_rotation: augmentation using rotation, linearly spaced from 0 deg to 90 deg inclusive
    n_oversample: augmentation multiplier for oap, obs (balancing tri-label data)
    datasets: list of dataset indices to use for train + eval
    train_split: ratio of training images
    train_datasets: list of dataset indices to use for training
    test_datasets: list of dataset indices to use for testing

    Note that train/test_datasets overrides datasets, train_split
    '''

    def cut_list(sorted_list, ratio):
        '''
        Get a tuple of two lists that each contain ratio, 1 - ratio of
        the original list
        '''
        n1 = int(len(sorted_list) * ratio)
        # n2 = len(sorted_list) - n1  # unused
        first = sorted_list[:n1]
        second = sorted_list[n1:]

        return first, second

    def save_augmented_images(src_path, dest_path, n_rotation, crop_ratio=0.2):
        def append_to_basename(jpg_path, append_string):
            if jpg_path[-4:] != '.jpg':
                raise ValueError("jpg_path does not conform to format '*.jpg'")

            return jpg_path[:-4] + append_string + '.jpg'

        angles = np.linspace(0, 90, n_rotation, dtype=int)

        if len(np.unique(angles)) != len(angles):
            raise Exception('Too many augmentations')

        loaded_image = plt.imread(src_path)

        for angle in angles:
            augmented_path = append_to_basename(
                dest_path, '_CP{:02d}_R{:03d}.aug'.format(int(crop_ratio * 100), angle))

            lx, ly = loaded_image.shape
            cropped_image = loaded_image[int(lx * crop_ratio): int(- lx * crop_ratio),
                                         int(ly * crop_ratio): int(- ly * crop_ratio)]

            rotated_image = scipy.ndimage.interpolation.rotate(
                cropped_image, angle)

            imageio.imsave(augmented_path, rotated_image)

    print('Saving images...')
    metadata = load_metadata()

    # train_split patients
    if train_datasets is None and test_datasets is None:
        used_datasets = datasets
        pids = []
        for cr_code in metadata:
            cr = parse_cr_code(cr_code)
            if cr[0] in datasets:
                pids.append(cr[1])
        pids = sorted(list(set(pids)))
        train_pids, test_pids = cut_list(pids, train_split)
    elif train_datasets is None or test_datasets is None:
        raise Exception(
            'train_datasets and test_datasets must be passed in conjuction')
    else:
        used_datasets = train_datasets + test_datasets
        train_pids = []
        test_pids = []
        for cr_code in metadata:
            cr = parse_cr_code(cr_code)
            if cr[0] in train_datasets:
                train_pids.append(cr[1])
            if cr[0] in test_datasets:
                test_pids.append(cr[1])
        train_pids = sorted(list(set(train_pids)))
        test_pids = sorted(list(set(test_pids)))
        pids = train_pids + test_pids

    print('Training Patients: {} / Testing Patients: {}'.format(len(train_pids), len(test_pids)))

    # init dataset info DataFrame
    if tri_label:
        semi_columns = ['OAP', 'IN', 'OBS', 'TOTAL']
    else:
        semi_columns = ['OAP', 'AP', 'MD', 'BS', 'OBS', 'TOTAL']
    columns = pd.MultiIndex.from_product([['Original', 'Augmented'], semi_columns])
    train_df = pd.DataFrame(np.zeros((3, 8)),
                      columns=columns,
                      index=('Image Count', 'Percentages', 'Images Per Patient'))
    test_columns = pd.Index(semi_columns)
    test_df = pd.DataFrame(np.zeros((3, 4)),
                      columns=test_columns,
                      index=('Image Count', 'Percentages', 'Images Per Patient'))

    # count total # of cr_codes to consider (for loading bar)
    count = len([None for cr_code in metadata if parse_cr_code(
        cr_code)[0] in used_datasets])
    bar = progress.bar.IncrementalBar('Copying Images...', max=count)

    # check IMAGES_DIR
    if glob.glob(os.path.join(IMAGES_DIR, '**/*.jpg'), recursive=True):
        print('WARNING: there are existing images in "{}"'.format(IMAGES_DIR))

    # copy images
    unlabeled = []
    done = []
    for cr_code, info in metadata.items():
        cr = parse_cr_code(cr_code)
        if cr[0] not in used_datasets:
            continue
        bar.next()
        done.append(cr_code)

        try:
            label = info['label']
        except KeyError:
            unlabeled.append(cr_code)
            label = 'in'  # hotfix: TODO fix cr_learn to accept non-labeled test data

        if tri_label and label in ['ap', 'md', 'bs']:
            label = 'in'

        if cr[1] in train_pids:
            dest = 'training_{}.jpg'.format(cr_code)
        else:
            dest = 'testing_{}.jpg'.format(cr_code)

        dest = os.path.join(label, dest)
        dest = os.path.join(IMAGES_DIR, dest)
        src = os.path.join(DATABASE_DIR, cr_code + '.jpg')
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if cr[1] in train_pids:
            if label == 'in':
                count = n_rotation
            else:
                count = n_rotation * n_oversample
            save_augmented_images(src, dest, count)
            train_df[('Original', label.upper())] += 1
            train_df[('Augmented', label.upper())] += count
        else:
            shutil.copy(src, dest)
            test_df[label.upper()] += 1
    bar.finish()

    # dataset summary overview
    total = train_df.loc['Image Count', 'Original'].sum()
    for i in range(3):
        train_df.loc['Percentages',columns[i]] = train_df.loc['Image Count', columns[i]] / total * 100
    total = train_df.loc['Image Count', 'Augmented'].sum()
    for i in range(4, 7):
        train_df.loc['Percentages',columns[i]] = train_df.loc['Image Count', columns[i]] / total * 100
    train_df.loc['Images Per Patient'] = train_df.loc['Image Count'] / len(train_pids)
    for label in ['Original', 'Augmented']:
        train_df.loc[:,(label, 'TOTAL')] = train_df.loc[:, label].sum(axis=1)

    total = test_df.loc['Image Count'].sum()
    test_df.loc['Percentages'] = test_df.loc['Image Count'] / total * 100
    test_df.loc['Images Per Patient'] = test_df.loc['Image Count'] / len(test_pids)
    test_df['TOTAL'] = test_df.sum(axis=1)

    # save dataset info to csvs
    train_df.to_csv('train_' + spec_csv)
    test_df.to_csv('test_' + spec_csv)
    print('saved train dataset info to train_' + spec_csv)
    print('saved test dataset info to test_' + spec_csv)

    # print copied images
    print('Copied')
    print(done)
    print('Unlabeled')
    print(unlabeled)


def visualize_metadata():
    metadata = load_metadata()
    print(json.dumps(metadata, sort_keys=True, indent=4, separators=(',', ': ')))


def load_results(results_dir=RESULTS_DIR):
    '''
    Returns
    [
        {
            'tfhub_module': url of tfhub module
            'training_steps': int
            'learning_rate': float
            'validation_percentage': float
            'batch_size': int
            'test_accuracy': float
            'training_images': [paths]
            'predictions': {
                'image_basename': {
                        'prediction': 'oap',
                        'truth': 'oap',
                        'percentages': {'oap': float (0-1)...}
                }, ...
            }
        }, ...
    ]
    '''
    result_paths = glob.glob(os.path.join(results_dir, '**/results.json'))
    results = []

    for path in result_paths:
        with open(path) as f:
            try:
                result = json.load(f)
            except json.decoder.JSONDecodeError as e:
                print('invalid results file: {}'.format(path))
                pass

        # typo fix
        if 'test_accuaracy' in result:
            result['test_accuracy'] = result['test_accuaracy']
            del result['test_accuaracy']

        results.append(result)
        json.dump(result, open(path, 'w'))

    return results


def select_result(results, sort_by=['test_accuracy']):
    def sort_key(result):
        values = []
        for key in sort_by:
            values.append(result[key])
        return values
    results.sort(key=sort_key, reverse=True)

    print('{:-^80}'.format(' Predictions List '))
    for i, result in enumerate(results):
        print('%d.\tModule: %s' % (i, result['tfhub_module']))
        print('\tSteps: %-10sRate: %-10sAccuracy: %-10s' % (
            result['training_steps'],
            result['learning_rate'],
            result['test_accuracy'])
        )
        print()

    while True:
        try:
            index = int(
                input('Which of the predictions would you like to use? '))
            return results[index]
        except (IndexError, ValueError):
            print('Invalid index')
            continue


def load_best_result(results_dir=RESULTS_DIR):
    results = load_results(results_dir)
    results.sort(key=lambda r: r['test_accuracy'])
    return results[-1]


def prompt_and_load_result(results_dir=RESULTS_DIR):
    results = load_results(results_dir)
    result = select_result(results)
    return result


def load_result(results_dir=RESULTS_DIR):
    # deprecated
    return prompt_and_load_result(results_dir)


def is_tri_label_result(result):
    for image_dict in result['predictions'].values():
        if image_dict['truth'] == 'ap':
            return False
    return True


def main():
    # visualize_metadata()
    # save_training_data()
    # save_training_data()

    #prepare_images(tri_label=True, n_rotation=6, n_oversample=1,
    #               train_datasets=[0], test_datasets=[3])
    prepare_images(tri_label=True, n_rotation=6, n_oversample=5,
                   datasets=[0])


if __name__ == "__main__":
    main()
