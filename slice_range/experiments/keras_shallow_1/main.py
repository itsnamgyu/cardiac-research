
import os
import sys
import random
import json
sys.path.append('../..')

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K

import cr_interface as cri
import keras_utils as ku

splits = cri.DATA_DIRS.keys()
application = ku.applications['vgg19']

SEED = 37
def reset_random():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(SEED)
    random.seed(SEED)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(SEED)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
def get_generators():
    transform_parameters = {
        'zx': 0.6,
        'zy': 0.6,
    }
    zoom_gen = ImageDataGenerator()
    zoom = lambda x: zoom_gen.apply_transform(x, transform_parameters)

    generators = dict()
    for split in splits:
        if split == 'test':
            augment_kwargs = dict()
        else:
            augment_kwargs = dict(
                rotation_range=45,
                fill_mode='nearest'
            )

        generators[split] = ImageDataGenerator(
            **augment_kwargs,
            preprocessing_function=zoom)

    return generators

def get_iterators(image_size):
    generators = get_generators()

    iterators = dict()

    kwargs = dict(
        target_size=image_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
        seed=SEED)

    for split, gen in generators.items():
        iterators[split] = gen.flow_from_directory(
            directory=cri.DATA_DIRS[split],
            **kwargs)
        
    return iterators

def get_labels(iterator, multiplier=1):
    # reset seed parameters
    # note that you need to use the same iterator to reproduce order
    iterator.total_batches_seen = 0
    iterator.batch_index = 0
    
    labels = None
    for i, batch in enumerate(iterator):
        if i == int(len(iterator) * multiplier):
            break
        if labels is None:
            labels = np.array(batch[1])
        else:
            labels = np.append(labels, np.array(batch[1]), axis=0)
            
    return labels

def get_filenames(iterator, multiplier=1):
    return np.tile(np.array(iterator.filenames), multiplier)

def generate_and_save_bottlenecks(application, multiplier=5):
    # create bottlenecks & save
    reset_random()
    iterators = get_iterators(application.image_size)
    bottlenecks = dict()
    labels = dict()
    filenames = dict()

    kwargs = dict(
        verbose=1,
        workers=8,
        use_multiprocessing=True)

    for split, it in iterators.items():
        bottlenecks[split] = application.get_model().predict_generator(
            it, steps=len(it) * multiplier, **kwargs)
        labels[split] = get_labels(it, multiplier=multiplier)
        filenames[split] = get_filenames(it, multiplier=multiplier)

    for split, data in bottlenecks.items():
        path = 'bottlenecks/{}/{}'.format(application.codename, split)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(open(path, 'wb'), data)

    for split, data in labels.items():
        path = 'labels/{}/{}'.format(application.codename, split)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(open(path, 'wb'), data)
        
    for split, data in filenames.items():
        path = 'filenames/{}/{}'.format(application.codename, split)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(open(path, 'wb'), data)

def load_all_bottlenecks():
    for key, app in ku.applications.items():
        app.free_model()
        generate_and_save_bottlenecks(app, multiplier=1)

def load_bottlenecks(application):
    bottlenecks = dict()
    labels = dict()
    filenames = dict()
    
    for split in splits:
        bottlenecks[split] = np.load(open('bottlenecks/{}/{}'.format(application.codename, split), 'rb'))
        labels[split] = np.load(open('labels/{}/{}'.format(application.codename, split), 'rb'))
        filenames[split] = np.load(open('filenames/{}/{}'.format(application.codename, split), 'rb'))

    return bottlenecks, labels, filenames


def compile_model(model, lr=1.0e-4):
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd,
        #optimizer='rmsprop',
        metrics=['accuracy'])
    
def load_top_model(app, compiled=True, lr=1.0e-4):
    model = Sequential()
    model.add(Flatten(input_shape=app.get_model().output_shape[1:]))
    model.add(Dense(256,
                        activation='relu',
                        kernel_initializer=keras.initializers.glorot_uniform(seed=SEED)))
    model.add(Dropout(0.5,
                         seed=SEED))
    model.add(Dense(3, 
                        activation='softmax',
                        kernel_initializer=keras.initializers.glorot_uniform(seed=SEED)))

    if compiled:
        compile_model(model, lr)
    
    return model

def train_model(model, bottlenecks, labels, tensorboard_name=None, epochs=10, batch_size=32):
    reset_random()
    
    if tensorboard_name:
        os.makedirs('boards', exist_ok=True)
        tensorboard = TensorBoard(log_dir="boards/{}".format(tensorboard_name))
        callbacks=[tensorboard]
    else:
        callbacks=None
        
    print('loaded board. now fitting...')
    model.fit(bottlenecks['train'], labels['train'],
                  validation_data=(bottlenecks['validation'], labels['validation']),
                  shuffle=True,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=callbacks)
    
DEFAULT_LABELS = ['in', 'oap', 'obs']
def test_model_and_report(model, bottlenecks, labels, filenames, label_names=DEFAULT_LABELS):
    '''
    Returns
    Accuracy & prediction part of cr_result.json as dict
    '''
    predictions = model.predict(bottlenecks['test'])

    results = dict()
    results['predictions'] = {}
    answers = 0
    for i, p in enumerate(predictions):
        max_prob = 0
        max_index = 0
        for j, prob in enumerate(p):
            if prob > max_prob:
                max_prob = prob;
                max_index = j;
        
        d = dict()
        d['truth'] = label_names[list(labels['test'][i]).index(1)]
        d['prediction'] = label_names[max_index]
        d['percentages'] = dict(zip(label_names, list(map(lambda f: str(f), p))))
        
        cr_code = cri.extract_cr_code(filenames['test'][i])
        results['predictions'][cr_code] = d
        
        if d['prediction'] == d['truth']:
            answers += 1
            
    results['test_accuracy'] = str(answers / len(predictions))
    
    return results

app = ku.applications['mobilenet']
all_results = list()
# app, lr, accuracy

for app in ku.applications.values():
    if app.name not in ['inceptionresnetv2', 'nasnet', 'resnet50']:
        continue
    bottlenecks, labels, filenames = load_bottlenecks(app)
    for lr_factor in [2, 3, 4, 5, 6, 7]:
        epochs = 25
        name = '{}_LR{}_E{:03d}'.format(app.codename, lr_factor, epochs)
        model = load_top_model(app, lr=0.1 ** lr_factor)
        train_model(model, bottlenecks, labels, epochs=epochs, tensorboard_name=name)
        results = test_model_and_report(model, bottlenecks, labels, filenames)
        
        params = dict()
        params['epochs'] = epochs
        params['lr'] = str(0.1 ** lr_factor)
        results['params'] = params
        
        results['description'] = \
'''Shallow tuning top layer with Keras.
Used pre-loaded bottlenecks with an augment factor of 6.
Top model: flatten - dense 256 - dropout 0.5 - softmax 3.
Trained on sep 5th.'''
        results['short_name'] = name
        
        results_path = 'results/{}/cr_result.json'.format(name)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        json.dump(results, open(results_path, 'w'))
        
        all_results.append((app.codename, lr_factor, results['test_accuracy']))
    
print(all_results)

with open('all_results', 'w') as f:
    for line in all_results:
        f.write(str(line) + '\n')
