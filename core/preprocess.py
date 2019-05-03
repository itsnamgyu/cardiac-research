import os
import glob
from tqdm import tqdm
import scipy
from scipy import ndimage
import numpy as np

import core


DATABASE_DIR = os.path.join(core.PROJECT_DIR, 'data', 'database')
PROCESSED_DIR = os.path.join(core.PROJECT_DIR, 'data', 'database_processed')


def preprocess_dataset(input_dir=DATABASE_DIR, output_dir=PROCESSED_DIR):
    os.makedirs(output_dir, exist_ok=True)
    input_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    print('Preprocessing data')
    for input_path in tqdm(input_paths):
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        if os.path.exists(output_path):
            continue

        image = ndimage.imread(input_path, mode='RGB')
        width, height, _ = image.shape
        maximum = max(width, height)

        d_width = maximum - width
        d_height = maximum - height
        pad_x1 = int(d_width / 2)
        pad_x2 = pad_x1 + d_width % 2
        pad_y1 = int(d_height / 2)
        pad_y2 = pad_y1 + d_height % 2

        pad = ((pad_x1, pad_x2), (pad_y1, pad_y2), (0, 0))
        result = np.pad(image, pad, mode='constant', constant_values=0)

        scipy.misc.imsave(output_path, result)


if __name__ == '__main__':
    preprocess_dataset()
