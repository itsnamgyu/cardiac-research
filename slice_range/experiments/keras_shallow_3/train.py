
# coding: utf-8

import sys
sys.path.append('../..')

import os
import json

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm

import cr_interface as cri
import keras_utils as ku
import keras_bottle as kb

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

def save_history(res: keras.callbacks.History, name):
    HIST_DIR = 'history'
    os.makedirs(HIST_DIR, exist_ok=True)
    hist_path = os.path.join(HIST_DIR,'{}.json'.format(name))
    with open(hist_path, 'w') as f:
        json.dump(res.history, f)


# In[39]:


def train_top_models(train_collection: cri.CrCollection, app):
    '''
    Train top models using K-fold & learning rate search
    '''
    SPLITS = 5
    OUT_MULTIPLIER = 6
    AUG_MULTIPLIER = 1
    LR_RANGE = [3, 4, 5, 6]
    EPOCHS = [10, 10, 100, 100]
    
    splits = train_collection.split_by(['dataset_index', 'pid'], [0.2] * 5)

    split_labels = []
    split_aug_labels = []
    split_bottles = []
    split_aug_bottles = []

    print('loading bottlenecks... ', end='')
    for i, split in enumerate(splits):
        #print('Loading split {} of {}...'.format(i + 1, len(splits)))
        cr_codes = split.get_cr_codes()
        labels = split.get_labels()
        bottles = kb.load_bottlenecks(
            cr_codes, app.codename, app.get_model(),
            augmented=False, multiplier=1, verbose=0)
        split_bottles.append(bottles)
        split_labels.append(labels)

        labels = []
        cr_codes = split.filter_by(label='in').get_cr_codes()
        labels += split.filter_by(label='in').get_labels()
        in_bottles = kb.load_bottlenecks(
            cr_codes, app.codename, app.get_model(),
            augmented=True, multiplier=AUG_MULTIPLIER, verbose=0)

        cr_codes = split.filter_by(label=['oap', 'obs']).get_cr_codes()
        labels += split.filter_by(label=['oap', 'obs']).get_labels()* OUT_MULTIPLIER
        out_bottles = kb.load_bottlenecks(
            cr_codes, app.codename, app.get_model(),
            augmented=True, multiplier=AUG_MULTIPLIER * OUT_MULTIPLIER,
            verbose=0)

        bottles = np.concatenate((in_bottles, out_bottles))

        split_aug_bottles.append(bottles)
        split_aug_labels.append(labels)
    print('done')
        
    labelize = LabelEncoder().fit_transform
    onehot = OneHotEncoder(sparse=False).fit_transform
    encode = lambda l: onehot(labelize(l).reshape(-1, 1))
    
    results = {}
    
    print('training models by split + learning rate')
    with tqdm(total=SPLITS * len(LR_RANGE)) as bar:
        for i in range(SPLITS):
            validation_labels = encode(split_labels[i])
            validation_bottles = split_bottles[i]
            train_labels = []
            train_bottles = []
            for j in list(range(0, i)) + list(range(i + 1, SPLITS)):
                train_labels += split_aug_labels[j]
                train_bottles.append(split_aug_bottles[j])
            train_labels = encode(train_labels)
            train_bottles = np.concatenate(train_bottles)

            for lr_factor, epochs in zip(LR_RANGE, EPOCHS):
                lr = 0.1 ** lr_factor
                top_model = load_top_model(app, lr=lr)

                res = top_model.fit(train_bottles, train_labels,
                                    validation_data=(validation_bottles, validation_labels),
                                    verbose=0, shuffle=True, epochs=epochs)

                name = '{}_LR{}_E{:03d}_S{:02d}'.format(
                    app.codename, lr_factor, epochs, i)
                save_history(res, name)
                
                MODEL_DIR = 'model'
                os.makedirs(MODEL_DIR, exist_ok=True)
                top_model.save(os.path.join(MODEL_DIR, '{}.hdf5'.format(name)))
                
                results[name] = res

                bar.update(1)
                
    for name, res in results.items():
        print(name, max(res.history['val_acc']))
    
    return results

collection = cri.CrCollection.load().labeled().tri_label()
train = collection.filter_by(dataset_index=0)
test = collection.filter_by(dataset_index=1)

all_results = []

for key, app in ku.applications.items():
    print(key.center(80, '-'))
    results = train_top_models(train, app)
    all_results.append(results)
    
with open('all_results.json') as f:
    json.dump(all_results, f)
