
# coding: utf-8

# In[24]:


import os

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.np_utils import to_categorical   
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras_applications import vgg16, vgg19, inception_v3, resnet50, mobilenet, mobilenet_v2, inception_resnet_v2, xception, densenet, nasnet
from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd


# In[2]:


IMAGE_SHAPE = (160, 160)


# In[3]:


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
    
    aug_gens = dict()
    aug_gens['train'] = ImageDataGenerator(
            rotation_range=40,
            fill_mode='nearest',
            preprocessing_function=zoom)
    aug_gens['validation'] = ImageDataGenerator(
            rotation_range=40,
            fill_mode='nearest',
            preprocessing_function=zoom)
    aug_gens['test'] = ImageDataGenerator(
            preprocessing_function=zoom)
    
    # get generator per dataset
    ordered_gens = dict()
    kwargs = dict(
        target_size=image_shape,
        batch_size=1,
        class_mode=None,
        shuffle=False
    )
    for key, aug_gen in aug_gens.items():
        ordered_gens[key] = aug_gen.flow_from_directory(
            '../data/data/{}'.format(key), **kwargs)
    
    # generate bottleneck labels after augmentation
    labels = dict()
    for key, gen in ordered_gens.items():
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
    for key, gen in ordered_gens.items():
        print('Preparing {} bottlenecks'.format(key))
        bottlenecks[key] = model.predict_generator(
            gen, steps=len(labels[key]), **kwargs
        )

    return bottlenecks, labels


# In[4]:


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


# In[7]:


pre_model = mobilenet.MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=get_input_shape(IMAGE_SHAPE)
)


# In[40]:


def train_and_eval(pre_model_func, image_shape, name='tune'):
    pre_model = pre_model_func(
        weights='imagenet',
        include_top=False,
        input_shape=get_input_shape(image_shape)
    )
    
    bottlenecks, labels = generate_bottlenecks_and_labels(
        pre_model, image_shape, augment_factor=5)

    for key, val in bottlenecks.items():
        np.save(open('bottlenecks_{}_{}.npy'.format(name, key), 'wb'), val)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=pre_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(3, activation='softmax'))
    top_model.compile(loss='categorical_crossentropy',
                      #optimizer=optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                      #optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    # one-hot labels
    hot_labels = dict()
    for key, label_array in labels.items():
        hot_labels[key] = to_categorical(label_array, num_classes=3)

    os.makedirs('logs', exist_ok=True)
    tensorboard = TensorBoard(log_dir="logs/{}".format(name))
    
    # train model
    batch_size = 16
    top_model.fit(bottlenecks['train'], hot_labels['train'],
                  validation_data=(bottlenecks['validation'], hot_labels['validation']),
                  epochs=25,
                  batch_size=batch_size,
                  shuffle=True,
                  callbacks=[tensorboard])

    # evaluate model
    results = top_model.evaluate(bottlenecks['test'], hot_labels['test'])

    # save weights for model
    top_model.save_weights('weights_{}.h5'.format(name))
    
    print(name, results)


# In[41]:


models = [
    (mobilenet.MobileNet, (224, 224), 'mobilenet'),
    (mobilenet_v2.MobileNetV2, (224, 224), 'mobilenetv2'),
    (inception_resnet_v2.InceptionResNetV2, (299, 299), 'inceptionresnetv2'),
    (inception_v3.InceptionV3, (299, 299), 'inceptionv3'),
    #(densenet.DenseNet, (224, 224), 'densenet'),
    (nasnet.NASNet, (224, 224), 'nasnet'),
    (resnet50.ResNet50, (224, 224), 'resnet50'),
    (vgg16.VGG16, (224, 224), 'vgg16'),
    (vgg19.VGG19, (224, 244), 'vgg19'),
    (xception.Xception, (299, 299), 'xception'),
]


# In[ ]:


#train_and_eval(mobilenet.MobileNet, (224, 224), 'mobilenet')
for model, shape, name in models:
    train_and_eval(model, shape, name)


'''
# In[14]:


for layer in pre_model.layers[:-3]:
    layer.trainable = False


# # Two Layer

# In[15]:


pre_model = mobilenet.MobileNet(
    weights='imagenet',
    include_top=False,
    input_shape=get_input_shape(IMAGE_SHAPE)
)


# In[16]:


for layer in pre_model.layers[-3:]:
    layer.trainable = True
for layer in pre_model.layers[:-3]:
    layer.trainable = False


# In[17]:


new_model = Sequential()
new_model.add(pre_model)


# In[18]:


top_model = Sequential()
top_model.add(Flatten(input_shape=pre_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(3, activation='softmax'))

new_model.add(top_model)


# In[19]:


top_model.load_weights('bottleneck_fc_model.h5')


# In[20]:


new_model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  #optimizer='rmsprop',
                  metrics=['accuracy'])


# In[21]:


batch_size = 16

transform_parameters = {
    'zx': 0.6,
    'zy': 0.6,
}
zoom_gen = ImageDataGenerator()
zoom = lambda x: zoom_gen.apply_transform(x, transform_parameters)
gen = ImageDataGenerator(
        preprocessing_function=zoom)
aug_gen = ImageDataGenerator(
        rotation_range=40,
        fill_mode='nearest',
        preprocessing_function=zoom)
aug_gen2 = ImageDataGenerator(
        rotation_range=40,
        fill_mode='nearest',
        preprocessing_function=zoom)

train_image_generator = aug_gen.flow_from_directory(
    '../data/data/train',
    target_size=(IMAGE_SHAPE),
    batch_size=batch_size)

validation_image_generator = aug_gen2.flow_from_directory(
    '../data/data/validation',
    target_size=(IMAGE_SHAPE),
    batch_size=batch_size)

test_image_generator = gen.flow_from_directory(
    '../data/data/test',
    target_size=(IMAGE_SHAPE),
    batch_size=batch_size,
    shuffle=False)


# In[22]:


current_epoch = 0


# In[23]:


epoch_count = 50
new_model.fit_generator(
    train_image_generator,
    epochs=epoch_count,
    validation_data=validation_image_generator,
    workers=8,
    use_multiprocessing=True)
current_epoch += epoch_count


# In[24]:


new_model.evaluate_generator(test_image_generator)


# In[25]:


new_model.predict_generator(train_image_generator)

'''
