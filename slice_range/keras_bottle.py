import os
import argparse
import tempfile

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import keras
import tqdm

import keras_utils as ku
import cr_interface as cri


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOTTLENECK_DIR = os.path.join(BASE_DIR, 'bottlenecks')


def get_bottleneck_dir(model_codename: str,
                       model: keras.models.Model,
                       mkdir=True):
    subdir = '{}_{}'.format(model_codename, len(model.layers))
    path = os.path.join(BOTTLENECK_DIR, subdir)
    
    if mkdir:
        os.makedirs(path, exist_ok=True)

    return os.path.abspath(path)

def load_bottlenecks(cr_codes, model_codename, model,
                     augmented=False, multiplier=5, generate=False,
                     verbose=1):
    bottle_dir = get_bottleneck_dir(model_codename, model)
    bottle_paths = []

    if augmented:
        for i in range(multiplier):
            for cr in cr_codes:
                bottle_paths.append(os.path.join(
                    bottle_dir, '{}_AUG_{}.npy'.format(cr, i)))
    else:
        for cr in cr_codes:
            bottle_paths.append(os.path.join(
                bottle_dir,'{}.npy'.format(cr)))

    loaded = 0
    for path in bottle_paths:
        if os.path.exists(path):
            loaded += 1

    bottles = []
    if loaded == len(bottle_paths):
        if verbose >= 1:
            print('loading bottlenecks...')
            for path in tqdm.tqdm(bottle_paths):
                bottles.append(np.load(path))
        else:
            for path in bottle_paths:
                bottles.append(np.load(path))
        return np.stack(bottles)
    else:
        if generate:
            raise NotImplementedError('image-wise generation not implemented')
        else:
            print('only generated {} of {} images'.format(
                loaded, len(bottle_paths)))

def generate_bottleneck(img_path, app, gen, model):
    '''
    Need to fix abstraction
    '''
    img = load_img(img_path)
    img = img.resize(app.image_size)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    kwargs = dict(
        batch_size=1,
        shuffle=False)

    flow = gen.flow(x, **kwargs)

    kwargs = dict(
        verbose=0,
        workers=8,
        use_multiprocessing=True)

    bottlenecks = model.predict_generator(flow, **kwargs)

    return bottlenecks[0]

def generate_bottlenecks(app, model, augment=False, out=False, multiplier=1):
    '''
    Need to fix abstraction
    '''
    bottleneck_dir = get_bottleneck_dir(app.codename, model)
    gen = app.get_image_data_generator(augment)

    collection = cri.CrCollection.load().tri_label()
    if out:
        collection = collection.filter_by(label=['obs', 'oap'])
    
    def batch(suffix=''):
        for cr_code in tqdm.tqdm(collection.get_cr_codes()):
            image_path = cri.get_image_path(cr_code)
            np_path = os.path.join(bottleneck_dir, '{}{}.npy'.format(cr_code, suffix))
            if not os.path.exists(np_path):
                bottle = generate_bottleneck(image_path, app, gen, model)
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    temp = f.name
                    np.save(f, bottle)
                os.rename(temp, np_path)

    if augment:
        for i in range(multiplier):
            print('generating augmented bottlenecks for {} ({}/{})'.format(
                app.codename, i + 1, multiplier))
            batch(suffix='_AUG_{}'.format(i))
    else:
        print('generating origin bottlenecks for {}'.format(app.codename))
        batch()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--multiplier', type=int, default=1)
    parser.add_argument('-O', '--out-multiplier', type=int, default=1)
    args = parser.parse_args()

    print('(1/3) Original Images')
    for app in ku.applications.values():
        generate_bottlenecks(app, app.get_model(), augment=False,
                             multiplier=args.multiplier)

    print('(2/3) Augmented')
    for app in ku.applications.values():
        generate_bottlenecks(app, app.get_model(), augment=True,
                             multiplier=args.multiplier)

    print('(3/3) Out Augmented')
    for app in ku.applications.values():
        generate_bottlenecks(app, app.get_model(), augment=True, out=True,
                             multiplier=args.multiplier * args.out_multiplier)

if __name__ == '__main__':
    main()
