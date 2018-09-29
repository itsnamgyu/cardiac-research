import os
import keras
import keras_utils


BOTTLENECK_DIR = 'bottlenecks'


def get_bottleneck_dir(model_codename: str,
                       model: keras.models.Model,
                       mkdir=True):
    subdir = '{}_{}'.format(model_codename, model.layers[-1].name)
    path = os.path.join(BOTTLENECK_DIR, subdir)
    
    if mkdir:
        os.makedirs(path, exist_ok=True)

    return os.path.abspath(path)
