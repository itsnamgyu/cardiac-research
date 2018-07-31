import os
import json
import shutil
from typing import Dict
import re
import glob
import argparse

DATABASE_DIR = 'cr_database'
METADATA_FILE = 'cr_metadata.json'
IMAGES_DIR = 'images'

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
    # load or initialize metadata
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

    print('Metadata file {} has been updated (including labels etc.)'.format(METADATA_FILE))


def get_cr_code(dataset_index, patient_index, phase_index, slice_index):
    cr_code = 'D%02d_P%08d_P%02d_S%02d' % (dataset_index, patient_index,
                                           phase_index, slice_index)
    return cr_code


re_cr_code = re.compile('D([0-9]{2})_P([0-9]{8})_P([0-9]{2})_S([0-9]{2})')


'''
Return cr_code from string that contains one
'''
def extract_cr_code(string):
    return re_cr_code.search(string).group(0)


'''
Return: (dataset_index, patient_index, phase_index, slice_index)
'''
def parse_cr_code(cr_code, match=True):
    if match:
        match = re_cr_code.match(cr_code)
    else:
        match = re_cr_code.search(cr_code)

    if not match:
        raise Exception('could not parse cr code {}'.format(cr_code))

    return tuple(map(lambda index: int(index), match.groups()))


# Temp Function
def save_training_data():
    metadata = load_metadata()

    for cr_code, info in metadata.items():
        cr = parse_cr_code(cr_code)
        if cr[0] == 0:
            if 'label' in info:
                dest = 'training_{}.jpg'.format(cr_code)
                dest = os.path.join(info['label'], dest)
                dest = os.path.join(IMAGES_DIR, dest)
        if cr[0] == 1:
            dest = 'testing_{}.jpg'.format(cr_code)
            dest = os.path.join('oap', dest)
            dest = os.path.join(IMAGES_DIR, dest)
        src = os.path.join(DATABASE_DIR, cr_code + '.jpg')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)


# Temp Function: save only images w/ labels (CAP training set)
def prepare_images(three_label=False):
    print('Saving test data.')
    metadata = load_metadata()

    pairs: Dict[str] = {}  # { patient_index: cr_code }

    for cr_code, info in metadata.items():
        cr = parse_cr_code(cr_code)
        if cr[0] == 0:
            pairs[cr[1]] = cr_code, info

    patient_indices = list(sorted(pairs.keys()))
    train_count = int(len(patient_indices) * 0.8)
    test_count = len(patient_indices) - train_count
    train_patients = patient_indices[:train_count]
    print('Training data: %d patients.' % train_count)
    print('Test data: %d patients.' % test_count)

    for cr_code, info in metadata.items():
        cr = parse_cr_code(cr_code)
        if cr[0] == 0 and 'label' in info:
            if cr[1] in train_patients:
                dest = 'training_{}.jpg'.format(cr_code)
            else:
                dest = 'testing_{}.jpg'.format(cr_code)
            label = info['label']
            if three_label:
                if label != 'obs' and label != 'oap':
                    label = 'in'
            dest = os.path.join(label, dest)
            dest = os.path.join(IMAGES_DIR, dest)
            src = os.path.join(DATABASE_DIR, cr_code + '.jpg')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(src, dest)


def visualize_metadata():
    metadata = load_metadata()
    print(json.dumps(metadata, sort_keys=True, indent=4, separators=(',', ': ')))


def load_results(results_dir='results'):
    result_paths = glob.glob(os.path.join(results_dir, '**/results.json'))
    results = []

    for path in result_paths:
        with open(path) as f:
            result = json.load(f)

        # typo fix
        if 'test_accuaracy' in result:
            result['test_accuracy'] = result['test_accuaracy']
            del result['test_accuaracy']

        results.append(result)
        json.dump(result, open(path, 'w'))

    return results


def select_result(results):
    print()
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
            index = int(input('Which of the predictions would you like to use? '))
            return results[index]
        except (IndexError, ValueError):
            print('Invalid index')
            continue


def load_result(results_dir='results'):
    results = load_results(results_dir)
    result = select_result(results)
    return result


def main():
    # visualize_metadata()
    # save_training_data()
    # save_training_data()
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--three_label', action='store_true')
    args = parser.parse_args()

    print(args.three_label)
    prepare_images(three_label=args.three_label)


if __name__ == "__main__":
    main()
