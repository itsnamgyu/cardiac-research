import os

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical   
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import vgg16, inception_v3, resnet50, mobilenet, inception_resnet_v2
import numpy as np
import pandas as pd


IMAGE_SHAPE = (150, 150)


def generate_bottlenecks_and_labels(model, image_shape, augment_factor):
    '''
    Self Explanatory
    
    Arguments
    - model: keras.model.Model, usually a pre-existing model excluding top-layers,
             with pre-trained weights
    - image_shape: tuple(x, y)
    - augment_factor: how many times to augment non-test images
    
    Returns a tuple of 2 elements
    - bottlenecks: dict of bottleneck np.ndarrays, by dataset
    - labels: dict of label np.ndarrays, by dataset
    '''
    
    # define augmenations
    transform_parameters = {
        'zx': 0.6,
        'zy': 0.6,
    }
    zoom_gen = ImageDataGenerator()
    zoom = lambda x: zoom_gen.apply_transform(x, transform_parameters)
    vanilla_gen = ImageDataGenerator(
            preprocessing_function=zoom)
    aug_gen = ImageDataGenerator(
            rotation_range=40,
            fill_mode='nearest',
            preprocessing_function=zoom)
    
    # get generator per dataset
    ordered_generators = dict()
    kwargs = dict(
        target_size=image_shape,
        batch_size=1,
        class_mode=None,
        shuffle=False
    )
    ordered_generators['train'] = aug_gen.flow_from_directory(
            '../data/data/train', **kwargs)
    ordered_generators['validation'] = aug_gen.flow_from_directory(
            '../data/data/validation', **kwargs)
    ordered_generators['test'] = vanilla_gen.flow_from_directory(
            '../data/data/test', **kwargs)
    
    # generate bottleneck labels after augmentation
    labels = dict()
    for key, gen in ordered_generators.items():
        if key == 'test':
            labels[key] = gen.classes
        else:
            labels[key] = np.tile(gen.classes, augment_factor)

    # generate bottlenecks by dataset
    kwargs = dict(
        verbose=1,
        workers=8,
        use_multiprocessing=True,
    )

    bottlenecks = dict()
    for key, gen in ordered_generators.items():
        print('Preparing {} bottlenecks'.format(key))
        bottlenecks[key] = model.predict_generator(
            gen, steps=len(labels[key]), **kwargs
        )

    return bottlenecks, labels


def get_input_shape(image_shape):
    '''
    Get input shape of conv-nets based on keras backend settings
    
    Returns
    tuple(n1, n2, n3)
    '''
    
    if keras.backend.image_data_format() == 'channels_first':
        return (3,) + image_shape 
    else:
        return image_shape + (3,)


inception = inception_resnet_v2.InceptionResNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=get_input_shape(IMAGE_SHAPE)
)

bottlenecks, labels = generate_bottlenecks_and_labels(
    inception, IMAGE_SHAPE, augment_factor=1)


top_model = Sequential()
top_model.add(Flatten(input_shape=inception.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='softmax'))
top_model.compile(loss='categorical_crossentropy',
                  #optimizer=optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                  #optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  optimizer='rmsprop',
                  metrics=['accuracy'])
top_model.trainable=True


hot_labels = dict()
for key, label_array in labels.items():
    hot_labels[key] = to_categorical(label_array, num_classes=3)


# train model
batch_size = 16
top_model.fit(bottlenecks['train'], hot_labels['train'],
              validation_data=(bottlenecks['validation'], hot_labels['validation']),
              epochs=50,
              batch_size=batch_size,
              shuffle=True)

top_model.evaluate(bottlenecks['test'], hot_labels['test'])
