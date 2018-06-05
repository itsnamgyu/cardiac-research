import glob
import os
import argparse
import scipy
import scipy.ndimage


def get_jpg_paths(base_directory='.'):
    return glob.glob(os.path.join(base_directory, '**/*.jpg'), recursive=True)


def append_to_basename(jpg_path, append_string):
    if jpg_path[-4:] != '.jpg':
        raise ValueError("jpg_path does not conform to format '*.jpg'")

    return jpg_path[:-4] + append_string + '.jpg'


augment_angles = [
    0, 355, 5,
    90, 95, 85,
    180, 185, 175,
    270, 275, 265,
]


def main():
    parser = argparse.ArgumentParser(description='Augment your jpg files')
    parser.add_argument('-d', '--directory', type=str, default='cap_labeled')
    parser.add_argument('-t', '--target', type=str, default='cap_augmented')
    parser.add_argument('-v', '--verbal', action='store_true')
    args = parser.parse_args()

    jpg_paths = get_jpg_paths(args.directory)
    jpg_count = len(jpg_paths)
    fives = 0

    print('augmenting %d images...' % jpg_count)

    for index, path in enumerate(jpg_paths):
        img_array = scipy.ndimage.imread(path)

        local_path = path.replace(args.directory + '/', '', 1)
        target_path = os.path.join(args.target, local_path)

        try:
            os.makedirs(os.path.dirname(target_path))
        except OSError:
            pass

        for angle in augment_angles:
            positive_angle = angle % 360
            rotated_image = scipy.ndimage.interpolation.rotate(
                img_array, positive_angle)
            augmented_path = append_to_basename(
                target_path, '_r%03d' % positive_angle)
            scipy.misc.imsave(augmented_path, rotated_image)

        if args.verbal:
            print('(%d/%d) %s' % (index + 1, jpg_count, local_path))
        else:
            current_percentage = (index + 1) / jpg_count * 100
            if fives != current_percentage // 5:
                fives = current_percentage // 5
                print('%.0f%% complete' % current_percentage)


if __name__ == '__main__':
    main()
