import sys
sys.path.append('../..')

import os
import json
import math

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

import cr_interface as cri
import cr_analysis as cra
import keras_utils as ku
import keras_bottle as kb
import keras_history as kh
import keras_weights as kw
import lib


def bottleneck_generator(bottlenecks, labels, n_aug, batch_size=32, shuffle=True):
    images_per_epoch = int(len(bottlenecks) / n_aug)
    batches_per_epoch = math.ceil(images_per_epoch / batch_size)

    batches_by_aug = list()

    for index in range(n_aug):
        bo, la = bottlenecks[index::n_aug], labels[index::n_aug]
        if shuffle:
            bo, la = sklearn.utils.shuffle(bo, la)
        batches = list()
        batches_by_aug.append(batches)
        for i in range(batches_per_epoch):
            i0 = i * batch_size
            i1 = i0 + batch_size
            batches.append((bo[i0:i1], la[i0:i1]))

    index = 0
    while True:
        for i in range(batches_per_epoch):
            yield batches_by_aug[index][i]
        index = (index + 1) % n_aug

TEST = False
VERBOSE = 1
LEARNING_RATES = [0.001, 0.0001, 0.00001]

train_collection = cri.CrCollection.load().filter_by(
    dataset_index=0).tri_label().labeled()
test_collection = cri.CrCollection.load().filter_by(
    dataset_index=1).tri_label().labeled()

def predict(app, test=TEST, verbose=VERBOSE, batch_size=32):
    if test:
        epochs = 5
        n_aug = 1
        train_collection = train_collection.sample(frac=0.1)
        train_index = 999
        k = 5
        learning_rates = LEARNING_RATES
    else:
        epochs = 50
        n_aug = 5
        train_index = 0
        k = 5
        learning_rates = LEARNING_RATES

    min_loss_by_lr = []
    min_loss_epoch_by_lr = []
    average_histories = []
    
    for lr_index, lr in enumerate(learning_rates):
        histories = []
        for split in range(5):
            histories.append(pd.DataFrame(kh.load_history(app.get_model(), lr_index, 50, split)))
        average_history = pd.concat(histories).groupby(level=0).mean()
        average_histories.append(average_history)
        min_loss_by_lr.append(average_history.val_loss.min())
        min_loss_epoch_by_lr.append(average_history.val_loss.idxmin())

    min_loss = pd.Series(min_loss_by_lr).min()
    m_lr_index = pd.Series(min_loss_by_lr).idxmin()
    m_acc = pd.Series(average_histories[m_lr_index].val_acc).min()
    m_epochs = min_loss_epoch_by_lr[m_lr_index]
    m_lr = learning_rates[m_lr_index]

    test_bottles, test_labels = kb.load_bottlenecks(app, test_collection, aug=False)

    top_model = app.load_top_model(lr=m_lr)
    kw.load_weights(top_model, train_index=0)
    res = top_model.fit_generator(
        bottleneck_generator(train_bottles, train_labels,
                             n_aug=n_aug, batch_size=batch_size, shuffle=True),
        steps_per_epoch=batches_per_epoch,
        epochs=m_epochs,
        verbose=0)

    predictions = top_model.predict(test_bottles)
    result = cra.Result.from_predictions(
        predictions, test_collection.get_cr_codes(),
        dict(epochs=m_epochs, lr=m_lr), 'shallow_{}'.format(app.codename),
        description='final results for 5-fold 3-lr optimized top-layer 181105'
    )
    result.to_json('shallow_final', app.codename)
    kb.reset_bottlenecks()
    
ku.run_for_all_apps(predict, title="Generating predications")
lib.notify('Done generating predication')
