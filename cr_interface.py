import os
import json
import shutil
from typing import Dict
import re
import glob
from collections import defaultdict
import warnings

import keras
from tqdm import tqdm
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(PROJECT_DIR, 'data/database')
DATASET_DIR = os.path.join(PROJECT_DIR, 'data/datasets')
METADATA_FILE = os.path.join(PROJECT_DIR, 'data/metadata.json')

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
TRAINED_WEIGHTS_DIR = os.path.join(OUTPUT_DIR, 'trained_weights')
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


class CrCollection:
    def __init__(self, df, copy=False):
        if copy:
            self.df = df.copy()
        else:
            self.df = df

    @classmethod
    def from_dict(cls, d):
        dict_of_series = defaultdict(list)
        dict_of_series['cr_code'] = list(d.keys())

        keys = ['label', 'original_name', 'original_filepath']
        for info in d.values():
            for key in keys:
                dict_of_series[key].append(info.get(key, ''))

        cr_keys = ['dataset_index', 'pid', 'phase_index', 'slice_index']
        for i, key in enumerate(cr_keys):
            for cr_code in dict_of_series['cr_code']:
                dict_of_series[key].append(parse_cr_code(cr_code)[i])

        index = ['cr_code'] + cr_keys + keys
        df = pd.DataFrame.from_dict(dict_of_series)[index]
        df.sort_values('cr_code')
        return cls(df)

    @classmethod
    def load(cls):
        '''
        Load all data from cr_metadata.json
        '''
        return cls.from_dict(load_metadata())

    def split_by(self, columns, ratios, seed=None, copy=False):
        cr_keys = ['dataset_index', 'pid', 'phase_index', 'slice_index']
        ratios = pd.Series(ratios)

        if ratios.sum() != 1:
            raise ValueError('sum of ratio values are not 1')

        if type(columns) == str:
            columns = [columns]  # hotfix string iteration issue

        for column in columns:
            if column not in cr_keys:
                raise ValueError('invalid column {}'.format(column))

        # build permutation of unique keys (e.g., database-patient pairs)
        keys = self.df.loc[:, columns].drop_duplicates()
        keys = keys.sort_values(columns)
        p_indices = np.random.RandomState(seed).permutation(keys.index)
        keys = keys.reindex(p_indices, copy=False)
        keys = keys.reset_index(drop=True)

        def filter_by_keys(df, keys):
            for column in columns:
                df = df.loc[df[column].isin(keys[column])]
            return df

        lower_bounds = pd.Series([0] + list(ratios)[:-1]).cumsum()
        upper_bounds = ratios.cumsum()
        splits = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            split_keys = keys.iloc[int(lower * len(keys)):int(upper *
                                                              len(keys))]
            df = self.df
            df = filter_by_keys(df, split_keys)
            df = df.reset_index(drop=True)
            splits.append(CrCollection(df, copy))

        return splits

    def k_split(self, k, seed=None, columns=['dataset_index', 'pid']):
        ratios = []
        for _ in range(k - 1):
            ratios.append(1 / k)
        ratios.append(1 - sum(ratios))
        return self.split_by(columns, ratios, seed)

    def filter_by(self, inplace=False, **kwargs):
        '''
        kwargs
        column_name: value or list_of_possible_values
        '''
        if inplace:
            df = self.df
        else:
            df = self.df.copy()

        for key, val in kwargs.items():
            try:
                df = df.loc[df[key].isin(val)]
            except TypeError:  # element
                df = df.loc[df[key] == val]

        df = df.sort_values('cr_code').reset_index(drop=True)

        if not inplace:
            return CrCollection(df)
        else:
            self.df = df

    def labeled(self, inplace=False):
        if inplace:
            df = self.df
        else:
            df = self.df.copy()

        df = df.loc[df['label'] != '']

        df = df.sort_values('cr_code').reset_index(drop=True)

        if not inplace:
            return CrCollection(df)
        else:
            self.df = df

    def tri_label(self, inplace=False):
        if inplace:
            df = self.df
        else:
            df = self.df.copy()

        df.loc[:, 'label'] = df.loc[:, 'label'].apply(to_tri_label)

        if not inplace:
            return CrCollection(df)
        else:
            self.df = df

    def get_cr_codes(self):
        return list(self.df.loc[:, 'cr_code'])

    def get_image_paths(self, generator=False):
        return get_image_paths(self.df['cr_code'], generator)

    def load_images(self, target_size=None, stack=False):
        '''
        Return list of np arrays representing each image
        Return stacked np array if stack=True
        '''
        if len(self.df) > 32:
            warnings.warn('Loading more than 32 images')

        images = []
        for path in self.get_image_paths(generator=True):
            image = keras.preprocessing.image.load_img(path,
                                                       target_size=target_size)
            images.append(image)

        if stack:
            return np.stack(images)
        else:
            return images

    def get_labels(self, generator=False):
        return list(self.df.loc[:, 'label'])

    def get_cr_codes_by_label(self):
        df = self.labeled(inplace=False).df
        labels = list(df.loc[:, 'label'].drop_duplicates())
        cr_codes = dict()

        for label in labels:
            cr_codes[label] = list(
                df.loc[df.loc[:, 'label'] == label]['cr_code'])

        return cr_codes

    def export_by_label(self, dest, balancing=5, verbose=0):
        _inner_labels = ['in', 'ap', 'md', 'bs']

        os.makedirs(dest, exist_ok=True)
        if not os.path.isdir(dest):
            raise OSError('export path already exists and is not a directory')

        if (self.df['label'] == '').any():
            warnings.warn('exporting by label ignores unlabeled images')
        labels = list(self.df.label.unique())
        for label in labels:
            sub_dest = os.path.join(dest, label)
            sub = self.filter_by(label=label)
            if label in _inner_labels:
                sub.export(sub_dest, n=1, verbose=verbose)
            else:
                sub.export(sub_dest, n=balancing, verbose=verbose)

    def export(self, dest, by_label=None, n=1, verbose=0):
        os.makedirs(dest, exist_ok=True)
        if not os.path.isdir(dest):
            raise OSError('export path already exists and is not a directory')

        if by_label is not None:
            warnings.warn(
                'The by_label argument in export is depreciated. Use export_by_label'
            )

        if by_label == True:
            self.export_by_label(dest, verbose)
        else:
            pairs = []
            for path in self.get_image_paths():
                for i in range(n):
                    base_path = os.path.basename(path)
                    dest_path = os.path.join(dest,
                                             '{:02d}_{}'.format(i, base_path))
                    pairs.append((path, dest_path))
            if verbose:
                for pair in tqdm(pairs):
                    shutil.copy(pair[0], pair[1])
            else:
                for pair in pairs:
                    shutil.copy(pair[0], pair[1])

    def sample(self, n=None, frac=None):
        return CrCollection(self.df.sample(n=n, frac=frac))

    def __add__(self, other):
        if isinstance(other, CrCollection):
            return CrCollection(pd.concat([self.df, other.df], copy=True))
        else:
            raise TypeError('cannot add CrCollection with {}'.format(
                type(other)))


def load_metadata() -> Dict[str, Dict[str, str]]:
    '''
    Load or initialize metadata file
    '''

    if os.path.isfile(METADATA_FILE):
        try:
            with open(METADATA_FILE) as f:
                metadata: Dict[str:Dict[str:str]] = json.load(f)
        except json.JSONDecodeError:
            raise Exception('corrupt metadata file: {}'.format(METADATA_FILE))
    else:
        print('no metadata file')
        print('initializing new metadata')
        metadata: Dict[str:Dict[str:str]] = {}

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


def get_image_path(cr_code):
    return os.path.join(DATABASE_DIR, '{}.jpg'.format(cr_code))


def get_image_paths(cr_codes, generator=False):
    if generator:
        return map(get_image_path, cr_codes)
    else:
        return list(map(get_image_path, cr_codes))


def visualize_metadata():
    metadata = load_metadata()
    print(
        json.dumps(metadata, sort_keys=True, indent=4, separators=(',', ': ')))


def to_tri_label(label):
    if label in ['ap', 'md', 'bs']:
        return 'in'
    else:
        return label


def get_label(cr_code, tri_label=True):
    metadata = load_metadata()
    try:
        label = metadata[cr_code]['label']
    except KeyError:
        label = None
    if tri_label:
        label = to_tri_label(label)
    return label


def get_labels(cr_codes, tri_label=True, generator=False):
    labels = map(lambda c: get_label(c, tri_label=tri_label), cr_codes)
    if not generator:
        labels = list(labels)
    return labels


def load_results(results_dir=RESULTS_DIR):
    warnings.warn('[cri.load_results] deprecated: refer to cr_analysis')
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
    warnings.warn('[cri.select_result] deprecated: refer to cr_analysis')

    def sort_key(result):
        values = []
        for key in sort_by:
            values.append(result[key])
        return values

    results.sort(key=sort_key, reverse=True)

    print('{:-^80}'.format(' Predictions List '))
    for i, result in enumerate(results):
        print('%d.\tModule: %s' % (i, result['tfhub_module']))
        print('\tSteps: %-10sRate: %-10sAccuracy: %-10s' %
              (result['training_steps'], result['learning_rate'],
               result['test_accuracy']))
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
    warnings.warn('[cri.load_best_result] deprecated: refer to cr_analysis')
    results = load_results(results_dir)
    results.sort(key=lambda r: r['test_accuracy'])
    return results[-1]


def prompt_and_load_result(results_dir=RESULTS_DIR):
    warnings.warn(
        '[cri.prompt_and_load_result] deprecated: refer to cr_analysis')
    results = load_results(results_dir)
    result = select_result(results)
    return result


def load_result(results_dir=RESULTS_DIR):
    warnings.warn(
        '[cri.load_result] deprecated: using cri.prompt_and_load_result')
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
