
# coding: utf-8

# In[1]:


import sys
sys.path.append('../..')
import os

import matplotlib as mpl
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

import cr_interface as cri
import keras_utils as ku


# In[19]:


app = ku.applications['vgg16']

def load_data(app):
    data = {
        'bottlenecks': dict(),
        'filenames': dict(),
        'labels': dict(),
    }
    
    splits = ['test', 'validation', 'train']
    
    for key, d in data.items():
        for split in splits:
            path = os.path.join(key, app.codename)
            path = os.path.join(path, split)
            d[split] = np.load(path)
    
    return data

data = load_data(app)


# In[3]:


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
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    if compiled:
        compile_model(model, lr)
    
    return model

def train_model(model, bottlenecks, labels, tensorboard_name=None, epochs=10, batch_size=32):
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


# In[28]:


'''
top_model = load_top_model(app)
train_model(top_model, data['bottlenecks'], data['labels'], epochs=10)


# In[ ]:


top_model.save('vgg16_top')


# In[20]:


top_model = keras.models.load_model('vgg16_top')


# In[21]:


top_model.evaluate(data['bottlenecks']['test'], data['labels']['test'])

'''

# In[12]:


batch_size = 16

train_image_generator = app.get_image_data_generator(augment=True).flow_from_directory(
    cri.DATA_DIRS['train'],
    target_size=app.image_size,
    batch_size=batch_size)

validation_image_generator = app.get_image_data_generator(augment=True).flow_from_directory(
    cri.DATA_DIRS['validation'],
    target_size=app.image_size,
    batch_size=batch_size)

test_image_generator = app.get_image_data_generator(augment=False).flow_from_directory(
    cri.DATA_DIRS['test'],
    target_size=app.image_size,
    batch_size=batch_size,
    shuffle=False)


# In[29]:


model = Sequential()
model.add(app.get_model())
for layer in model.layers[0].layers[:-2]:
    layer.trainable = False
top_model = keras.models.load_model('vgg16_top')
model.add(top_model)
compile_model(model)
res = model.evaluate_generator(train_image_generator, verbose=1,
                               workers=8, use_multiprocessing=True)
print(res)


# In[26]:


#for lr in [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7]:
for lr in [1.0e-6]:
    print('Learning Rate: {}'.format(lr).center(80, '-'))
    model = Sequential()
    model.add(app.get_model())
    for layer in model.layers[0].layers[:-2]:
        layer.trainable = False
    top_model = keras.models.load_model('vgg16_top')
    model.add(top_model)
    compile_model(model, lr=lr)

    model.fit_generator(
        train_image_generator,
        epochs=5,
        validation_data=validation_image_generator,
        workers=8,
        shuffle=True,
        use_multiprocessing=True)

    model.evaluate_generator(test_image_generator)
