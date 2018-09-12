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

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(PROJECT_DIR, 'data/database')
DATASET_DIR = os.path.join(PROJECT_DIR, 'data/datasets')
DATA_DIR = os.path.join(PROJECT_DIR, 'data/data')
METADATA_FILE = os.path.join(PROJECT_DIR, 'data/metadata.json')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
SPEC_CSV = os.path.join(PROJECT_DIR, 'analysis/images_spec.csv')

DATA_DIRS = {}
for split in ['test', 'validation', 'train']:
    DATA_DIRS[split] = os.path.join(DATA_DIR, split)


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
        shutil.copyfile(METADATA_FILE, METADATA_FILE + '.bak')

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


def extract_cr_codes(strings, generator=False):
    if generator:
        return map(lambda s: extract_cr_code(s), strings)
    else:
        return list(map(lambda s: extract_cr_code(s), strings))


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
    result_paths = glob.glob(os.path.join(results_dir, '**/cr_result.json'))
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
    pass


if __name__ == "__main__":
    main()
