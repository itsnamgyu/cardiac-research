import numpy as np
import os
import json
import cr_interface as cri

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

DATABASE_DIR = 'cr_database'
METADATA_FILE = 'cr_metadata.json'
IMAGES_DIR = 'images'


def load_labels(label_file='label_data.npy'):
    metadata = cri.load_metadata()

    labels = ['nan', 'oap', 'ap', 'md', 'bs', 'obs']
    ndarray = np.load(label_file)
    label_data = {}
    for i in range(ndarray.shape[0]):
        name = ndarray[i][0]  # nonext basename
        label = labels[int(ndarray[i][1])]
        label_data[name] = label

    for data in metadata.values():
        name = data['original_name']
        if name in label_data:
            # if 'label' in data:
            #    print('label already loaded for {}'.format(name))
            # else:
            data['label'] = label_data[name]

    cri.save_metadata(metadata)


def main():
    load_labels()


if __name__ == "__main__":
    main()
