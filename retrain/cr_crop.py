import glob
from functools import reduce
import numpy
import scipy.ndimage
import os


def append_to_basename(jpg_path, append_string):
    if jpg_path[-4:] != '.jpg':
        raise ValueError("jpg_path does not conform to format '*.jpg'")

    return jpg_path[:-4] + append_string + '.jpg'


def main():
    images = glob.glob('images/**/*.jpg')
    crop_ratio = 0.2

    if reduce(lambda result, path: result or '.aug.' in path, images, False):
        print('You must rescale before augmentation')
        return
        
    for path in images:
        print(path)
        image = scipy.ndimage.imread(path)
        lx, ly = image.shape
        cropped_image = image[int(lx * crop_ratio): int(- lx * crop_ratio), 
                int(ly * crop_ratio): int(- ly * crop_ratio)]
        os.remove(path)
        scipy.misc.imsave(append_to_basename(path, '_CP20'), cropped_image)


if __name__ == '__main__':
    main()
