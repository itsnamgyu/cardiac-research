'''
Module for saving and loading trained bottlenecks
'''

import os

import keras_utils as ku
import cr_interface as cri


WEIGHTS_DIR = os.path.join(cri.PROJECT_DIR, 'weights')
TEMP_DIR = os.path.join(cri.PROJECT_DIR, '.keras_weights_temp')


def _get_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


def _get_weights_dir(train_index=0):
    '''
    - train_index: index for training configuration
      change this whenever you change your training configs
    '''
    dirname = os.path.join(WEIGHTS_DIR, 'train_{:03d}'.format(train_index))
    os.makedirs(dirname, exist_ok=True)
    return dirname


def get_weights_path(model, train_index=0):
    dirname = _get_weights_dir(train_index)
    basename = '{}_{}.h5'.format(model.name.replace('.', '_'), len(model.layers))
    return os.path.join(dirname, basename)


def save_weights(model, train_index=0):
    model.save_weights(get_weights_path(model, train_index))


def load_weights(model, train_index=0):
    model.load_weights(get_weights_path(model, train_index))


def reset_weights(model, train_index=0):
    path = get_weights_path(model, train_index)
    if os.path.exists(path):
        os.remove(path)


def main():
    print('testing random mobilenet save/load on train_index 999')
    save_weights(ku.apps['mobilenet'].get_model(), train_index=999)
    load_weights(ku.apps['mobilenet'].get_model(), train_index=999)
    reset_weights(ku.apps['mobilenet'].get_model(), train_index=999)


if __name__ == '__main__':
    main()
