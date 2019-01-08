import os
import argparse
import tempfile
import warnings
import shutil
import argparse
from collections import defaultdict

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import keras
from tqdm import tqdm
import requests
import pandas as pd

import keras_utils as ku
import cr_interface as cri
import lib


BOTTLENECK_DIR = os.path.join(cri.PROJECT_DIR, 'bottlenecks')
TEMP_DIR = os.path.join(cri.PROJECT_DIR, '.keras_bottle_temp')


def _get_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def get_bottleneck_dir(model, mkdir=True):
    subdir = '{}_{}'.format(model.name.replace('.', '_'), len(model.layers))
    path = os.path.join(BOTTLENECK_DIR, subdir)
    if mkdir:
        os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def get_bottleneck_path(model, cr_code, aug, index=0):
    dirname = get_bottleneck_dir(model)
    if not aug and index != 0:
        warnings.warn('index ignored on aug=False')
    if aug:
        basename = '{}_AUG_{:d}.npy'.format(cr_code, index)
    else:
        basename = '{}.npy'.format(cr_code)
    return os.path.join(dirname, basename)


def bottleneck_exists(model, cr_code, aug: bool, index=0):
    # optimization considerations: currently takes 35s per 1,0000,000 calls
    return os.path.exists(get_bottleneck_path(model, cr_code, aug, index))


def _get_nth_bottleneck_collection(app, collection, aug, index=0, model=None):
    '''
    Filter the collection to leave out just the ones we need
    to generate bottlenecks for
    '''
    if len(collection.df) == 0:
        warnings.warn('empty collection')
        return collection

    if not model:
        model = app.get_model()

    def should_generate(cr_code):
        return not bottleneck_exists(model, cr_code, aug, index)
    df = collection.df
    df = df.loc[df.apply(lambda x: should_generate(x['cr_code']), axis=1)]
    df = df.sort_values('cr_code')

    return cri.CrCollection(df)


def load_bottlenecks(app, base_collection, aug, count=1,
                     model=None, verbose=1, generate_only=False):
    '''
    generate and load bottlenecks/labels in this order
    X_AUG0 >> ...  >> X_AUGN >> Y_AUG0 >> ... >> Y_AUGN >> Z...

    TODO
    change signature to generate=False, and create a separate
    generate_bottlenecks function
    '''
    if len(base_collection.df) == 0:
        warnings.warn('empty collection')
        return None, None
    if count != 1 and not aug:
        warnings.warn('count ignored when aug is false')
        count = 1
    if not model:
        model = app.get_model()

    n_generate = 0
    total = len(base_collection.df) * count
    collections = []
    for i in range(count):
        c = _get_nth_bottleneck_collection(app, base_collection, model=model,
                                           aug=aug, index=i)
        collections.append(c)
        n_generate += len(c.df)

    if verbose:
        print('generating {} of {} ({} * {}) bottlenecks'.format(
            n_generate, total, len(base_collection.df), count))

    for index, collection in enumerate(collections):
        if len(collection.df) == 0:
            continue

        temp = _get_temp_dir()
        image_dir = os.path.join(temp, 'images')
        shutil.rmtree(image_dir, ignore_errors=True)
        default_class = os.path.join(image_dir, 'default_class')

        collection.export(default_class, by_label=False)

        gen = app.get_image_data_generator(augment=aug).flow_from_directory(
            image_dir, target_size=app.image_size, shuffle=False, batch_size=128)

        data = model.predict_generator(gen)

        # prevent corrupt data in main bottleneck directory
        temp_bottle_path = os.path.join(temp, 'temp.npy')
        for i, cr_code in enumerate(collection.get_cr_codes()):
            path = get_bottleneck_path(model, cr_code, aug, index=index)
            np.save(temp_bottle_path, data[i])
            os.rename(temp_bottle_path, path)

    if not generate_only:
        if verbose:
            print('loading {} bottlenecks'.format(total))
        bottles = []
        labels = np.repeat(base_collection.get_labels(), count)

        with tqdm(total=total, disable=(verbose < 2)) as bar:
            for cr_code in base_collection.get_cr_codes():
                for i in range(count):
                    path = get_bottleneck_path(model, cr_code, aug, index=i)
                    bottles.append(np.load(path))
                    bar.update(1)
        return np.stack(bottles), labels


def generate_all_bottlenecks(app, collection=None, augmentation=5, balancing=5, verbose=2):
    if collection:
        c = collection.tri_label()
        if len(c.df) == 0:
            warnings.warn('empty collection')
            return
    else:
        c = cri.CrCollection.load().labeled().tri_label()

    c0 = c.filter_by(dataset_index=0)
    c1 = c.filter_by(dataset_index=1)

    c0_out = c0.filter_by(label=['oap', 'obs'])
    c0_in = c0.filter_by(label='in')
    c1_out = c1.filter_by(label=['oap', 'obs'])
    c1_in = c1.filter_by(label='in')

    if verbose >= 2:
        print('(1/3) generating unaugmented bottlenecks'.center(100, '-'))
    elif verbose:
        print('(1/3) generating unaugmented bottlenecks')
    load_bottlenecks(app, c, aug=False, generate_only=True, verbose=verbose-1)

    if verbose >= 2:
        print('(2/3) generating inner train bottlenecks'.center(100, '-'))
    elif verbose:
        print('(2/3) generating inner train bottlenecks')
    load_bottlenecks(app, c0_in, aug=True, count=augmentation,
                     generate_only=True, verbose=verbose-1)

    if verbose >= 2:
        print('(3/3) generating outer train bottlenecks'.center(100, '-'))
    elif verbose:
        print('(3/3) generating outer train bottlenecks')
    load_bottlenecks(app, c0_out, aug=True, count=augmentation *
                     balancing, generate_only=True, verbose=verbose-1)


def get_k_bottlenecks(app, base_collection, k=5, n_aug=1, n_balance=5, verbose=1, include_entire_set=False):
    '''
    MEMORY ISSUE

    Return
    k_train_sets, k_validation_sets
    - sets: [ (bottles, labels), ... ]
    '''
    base_collection = base_collection.tri_label().labeled()
    splits = base_collection.k_split(['dataset_index', 'pid'], k)

    if verbose >= 0:
        print('Number of images by split by label'.center(100, '-'))
        d = defaultdict(list)
        for split in splits:
            d['total'].append(len(split.df))
            for label in ['in', 'oap', 'obs']:
                d[label].append(len(split.filter_by(label=label).df))
        print(pd.DataFrame(d))

    if verbose:
        print('Generating bottlenecks'.center(100, '-'))
    generate_all_bottlenecks(app, base_collection, verbose=verbose)

    split_labels = []
    split_aug_labels = []
    split_bottles = []
    split_aug_bottles = []

    if verbose:
        print('Loading bottlenecks'.center(100, '-'))
    for i, split in enumerate(splits):
        print('Loading split {} of {}...'.format(i + 1, k))
        # non-aug
        bottles, labels = load_bottlenecks(
            app, split, aug=False, verbose=verbose-1)
        split_bottles.append(bottles)
        split_labels.append(labels)

        # aug
        c = split.filter_by(label='in')
        b_in, l_in = load_bottlenecks(
            app, c, aug=True, count=n_aug, verbose=verbose-1)
        c = split.filter_by(label=['oap', 'obs'])
        b_out, l_out = load_bottlenecks(
            app, c, aug=True, count=n_aug * n_balance, verbose=verbose-1)
        split_aug_bottles.append(np.concatenate((b_in, b_out)))
        split_aug_labels.append(np.concatenate((l_in, l_out)))

    k_train_sets = []  # [ (bottles, labels) ... ]
    k_validation_sets = []  # [ (bottles, labels) ... ]

    for i in range(k):
        labels = lib.onehot(split_labels[i])
        bottles = split_bottles[i]
        k_validation_sets.append((bottles, labels))

        labels = []
        bottles = []
        for j in list(range(0, i)) + list(range(i + 1, k)):
            labels.append(split_aug_labels[j])
            bottles.append(split_aug_bottles[j])
        bottles = np.concatenate(bottles)
        labels = np.concatenate(labels)
        labels = lib.onehot(labels)
        k_train_sets.append((bottles, labels))

    if include_entire_set:
        labels = []
        bottles = []
        for i in range(k):
            labels.append(split_aug_labels[i])
            bottles.append(split_aug_bottles[i])
        bottles = np.concatenate(bottles)
        labels = np.concatenate(labels)
        labels = lib.onehot(labels)
        entire_set = (bottles, labels)
        return k_train_sets, k_validation_sets, entire_set
    else:
        return k_train_sets, k_validation_sets


def get_k_bottleneck_splits(app, base_collection, k=5, n_aug=1, n_balance=5, verbose=1):
    '''
    HOTFIX
    Return
    aug_bottles, aug_labels, origin_bottles, origin_labels
    - need to pick 4 aug bottle/labels and 1 non-aug for each validation run in k-split
    '''
    base_collection = base_collection.tri_label().labeled()
    splits = base_collection.k_split(['dataset_index', 'pid'], k)

    if verbose >= 0:
        print('Number of images by split by label'.center(100, '-'))
        d = defaultdict(list)
        for split in splits:
            d['total'].append(len(split.df))
            for label in ['in', 'oap', 'obs']:
                d[label].append(len(split.filter_by(label=label).df))
        print(pd.DataFrame(d))

    if verbose:
        print('Generating bottlenecks'.center(100, '-'))
    generate_all_bottlenecks(app, base_collection, verbose=verbose)

    split_labels = []
    split_aug_labels = []
    split_bottles = []
    split_aug_bottles = []

    if verbose:
        print('Loading bottlenecks'.center(100, '-'))
    for i, split in enumerate(splits):
        print('Loading split {} of {}...'.format(i + 1, k))
        # non-aug
        bottles, labels = load_bottlenecks(
            app, split, aug=False, verbose=verbose-1)
        split_bottles.append(bottles)
        split_labels.append(labels)

        # aug
        c = split.filter_by(label='in')
        b_in, l_in = load_bottlenecks(
            app, c, aug=True, count=n_aug, verbose=verbose-1)
        c = split.filter_by(label=['oap', 'obs'])
        b_out, l_out = load_bottlenecks(
            app, c, aug=True, count=n_aug * n_balance, verbose=verbose-1)
        split_aug_bottles.append(np.concatenate((b_in, b_out)))
        split_aug_labels.append(np.concatenate((l_in, l_out)))

    return split_aug_bottles, split_aug_labels, split_bottles, split_labels


def compile_kth_set(i, k, split_aug_bottles, split_aug_labels,
                    split_bottles, split_labels):
    '''
    HOTFIX: use with get_k_bottleneck_splits
    Concat training/test set for kth validation run
    '''
    if i == -1:
        # return entire set
        test_labels = []
        test_bottles = []
        for j in range(k):
            test_labels.append(split_labels[j])
            test_bottles.append(split_bottles[j])
        test_bottles = np.concatenate(test_bottles)
        test_labels = np.concatenate(test_labels)
        test_labels = lib.onehot(test_labels)

        train_labels = []
        train_bottles = []
        for j in range(k):
            train_labels.append(split_aug_labels[j])
            train_bottles.append(split_aug_bottles[j])
        train_bottles = np.concatenate(train_bottles)
        train_labels = np.concatenate(train_labels)
        train_labels = lib.onehot(train_labels)
        return train_bottles, train_labels, test_bottles, test_labels
    else:
        if i not in list(range(k)):
            raise ValueError('invalid index')
        val_labels = lib.onehot(split_labels[i])
        val_bottles = split_bottles[i]

        train_labels = []
        train_bottles = []
        for j in list(range(0, i)) + list(range(i + 1, k)):
            train_labels.append(split_aug_labels[j])
            train_bottles.append(split_aug_bottles[j])
        train_bottles = np.concatenate(train_bottles)
        train_labels = np.concatenate(train_labels)
        train_labels = lib.onehot(train_labels)
        return train_bottles, train_labels, val_bottles, val_labels


def reset_bottlenecks():
    shutil.rmtree(BOTTLENECK_DIR, ignore_errors=True)
    os.makedirs(BOTTLENECK_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-T', '--test', action='store_true')
    group.add_argument('-R', '--reset', action='store_true')
    args = parser.parse_args()

    if args.test:
        string = input('reset bottlenecks before test? (yes/no): ')
        if string == 'yes':
            print('resetting..')
            reset_bottlenecks()
        print('testing bottleneck generation on mobilnet and inception')
        fc = cri.CrCollection.load().sample(frac=0.02).labeled().tri_label()
        generate_all_bottlenecks(ku.apps['mobilenet'], collection=fc,
                                 augmentation=5, balancing=5)
        generate_all_bottlenecks(ku.apps['inceptionresnetv2'], collection=fc,
                                 augmentation=5, balancing=5)
        if string == 'yes':
            print('resetting..')
            reset_bottlenecks()
        print('done')
    elif args.reset:
        string = input('reset all bottlenecks? (yes/no): ')
        if string == 'yes':
            reset_bottlenecks()
            print('done')
        else:
            print('skip')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()