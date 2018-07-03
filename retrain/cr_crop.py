import glob
from functools import reduce
import numpy
import scipy.ndimage


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
        scipy.misc.imsave(path, cropped_image)

if __name__ == '__main__':
    main()
