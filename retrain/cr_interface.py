import os
import json
import shutil
from typing import List, Dict
import re

DATABASE_DIR = 'cr_database'
METADATA_FILE = 'cr_metadata.json'
IMAGES_DIR = 'images'

'''
Metadata Dict
{
    cr_code: {
        database_index: int
        phase_index: int
        slice_index: int
        original_filepath: int
    }
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


def get_cr_code(dataset_index, patient_index, phase_index, slice_index):
    cr_code = 'D%02d_P%08d_P%02d_S%02d' % (dataset_index, patient_index,
                                           phase_index, slice_index)
    return cr_code


'''
Return: (dataset_index, patient_index, phase_index, slice_index)
'''


def parse_cr_code(cr_code):
    re_cr_code = re.compile('D([0-9]{2})_P([0-9]{8})_P([0-9]{2})_S([0-9]{2})')
    match = re_cr_code.match(cr_code)

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
            dest = os.path.join('nan', dest)
            dest = os.path.join(IMAGES_DIR, dest)
        src = os.path.join(DATABASE_DIR, cr_code + '.jpg')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(src, dest)


def visualize_metadata():
    metadata = load_metadata()
    print(json.dumps(metadata, sort_keys=True, indent=4, separators=(',', ': ')))


def main():
    # visualize_metadata()
    # save_training_data()
    save_training_data()


if __name__ == "__main__":
    main()
