
# coding: utf-8

# In[1]:


import os
import sys
import random
import json
import glob
import tempfile
sys.path.append('../..')

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import tqdm

import cr_interface as cri
import cr_analysis as cra
import keras_utils as ku
import keras_bottle as kb


# In[2]:


app = list(ku.applications.values())[0]
kb.get_bottleneck_dir(app.codename, app.get_model())


# In[3]:


splits = cri.DATA_DIRS.keys()

SEED = 37
def reset_random():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(SEED)
    random.seed(SEED)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(SEED)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
def get_generator(app, augment=True):
    transform_parameters = {
        'zx': 0.6,
        'zy': 0.6,
    }
    zoom_gen = ImageDataGenerator()
    zoom = lambda x: app.preprocess_input(zoom_gen.apply_transform(x, transform_parameters))

    if augment:
        augment_kwargs = dict(
            rotation_range=45,
            fill_mode='nearest'
        )
    else:
        augment_kwargs = dict()

    return ImageDataGenerator(
        **augment_kwargs,
        preprocessing_function=zoom)

def generate_bottleneck(app, gen, img_path):
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

    bottlenecks = app.get_model().predict_generator(flow, **kwargs)
    
    return bottlenecks[0]


# In[4]:


images = glob.glob('{}/**/*.jpg'.format(cri.DATABASE_DIR), recursive=True)
images[0]


# In[5]:


def generate_bottlenecks(app, augment=False, multiplier=1):
    app.free_model()
    bottleneck_dir = kb.get_bottleneck_dir(app.codename, app.get_model())
    images = glob.glob('{}/**/*.jpg'.format(cri.DATABASE_DIR), recursive=True)
    gen = get_generator(app, augment)
    
    def batch(suffix=''):
        for image in tqdm.tqdm(images):
            cr = cri.extract_cr_code(image)
            path = os.path.join(bottleneck_dir, '{}{}.npy'.format(cr, suffix))
            if not os.path.exists(path):
                bottle = generate_bottleneck(app, gen, image)
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    temp = f.name
                    np.save(f, bottle)
                os.rename(temp, path)
                
    if augment:
        for i in range(multiplier):
            print('generating augmented bottlenecks for {} ({}/{})'.format(
                app.codename, i + 1, multiplier))
            batch(suffix='_AUG_{}'.format(i))
    else:
        print('generating origin bottlenecks for {}'.format(app.codename))
        batch()


for app in ku.applications.values():
    generate_bottlenecks(app, augment=False, multiplier=1)

for app in ku.applications.values():
    generate_bottlenecks(app, augment=True, multiplier=5)
