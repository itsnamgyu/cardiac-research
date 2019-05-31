#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib as mpl

mpl.use(
    'Agg'
)  # don't display mpl windows (will cause error in non-gui environment)
from collections import defaultdict
import math
import os
import shutil
import pandas as pd
from bayes_opt import BayesianOptimization
import keras

import core.history as ch
import core.fine_model as cm
from core.fine_model import FineModel

import analysis
import cr_interface as cri
import keras_utils as ku
from lib import Timer, notify
import traceback

try:
    BATCH_SIZE = 32
    K = 5
    BALANCE = 5
    LEARNING_RATES = [0.001, 0.0001, 0.00001]
    EPOCHS = 100
    SAMPLE = False  # sample 10% of examples for testing (sanity check stage)

    # In[3]:

    TEMP_IMAGE_DIR = 'temp_image'

    def get_train_val_generators(fm: FineModel, folds):
        """
        Get train/validation ImageDataGenerators for the given model for each fold.
        Note that subsequent calls to this method will invalidate the generators
        returned from previous calls.
        
        Train/validation images are BOTH BALANCED AND AUGMENTED
        
        :param fm: 
        The base model for which you want to use the generators
        
        :param folds: 
        
        :return: 
        tuple(train_gens, val_gens)
        
        train_gens: list of ImageDataGenerators for the train data in each fold
        val_gens: list of ImageDataGenerators for the validation data in each fold
        """
        print('Loading Train/Val ImageDataGenerators'.center(80, '-'))

        aug_gen = fm.get_image_data_generator(augment=True)

        val_gens = []
        train_gens = []

        for i in range(len(folds)):
            val_dir = os.path.join(TEMP_IMAGE_DIR, 'val{}'.format(i))
            train_dir = os.path.join(TEMP_IMAGE_DIR, 'train{}'.format(i))

            # refresh directories
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)
            shutil.rmtree(val_dir)
            shutil.rmtree(train_dir)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)

            fold: cri.CrCollection
            for j, fold in enumerate(folds):
                if i == j:
                    # export validation data for fold i
                    fold.export_by_label(val_dir, balancing=5)
                else:
                    # export train data for fold i
                    fold.export_by_label(train_dir, balancing=5)

            train_gens.append(
                aug_gen.flow_from_directory(
                    train_dir,
                    target_size=fm.get_output_shape(),
                    batch_size=BATCH_SIZE,
                    class_mode='categorical',
                ))
            val_gens.append(
                aug_gen.flow_from_directory(
                    val_dir,
                    target_size=fm.get_output_shape(),
                    batch_size=BATCH_SIZE,
                    class_mode='categorical',
                ))

            print(
                'Fold {}: {:<4} train images / {:<4} validation images'.format(
                    i + 1,
                    train_gens[-1].n,
                    val_gens[-1].n,
                ))

        test_dir = os.path.join(TEMP_IMAGE_DIR, 'test')
        for fold in folds:
            # export test data for all
            fold.export_by_label(test_dir, balancing=1)

        return train_gens, val_gens

    def get_test_generator(fm: FineModel, test_collection: cri.CrCollection):
        """
        Get ImageDataGenerator for the test data, compatible with the given model.
        Note that subsequent calls to this method will invalidate the generator
        returned from previous calls.
        
        Test images are NOT AUGMENTED NOR BALANCED
        
        :param fm: 
        The base model for which you want to use the generators
        
        :param test_collection:
        CrCollection containing test data
        
        :return: 
        ImageDataGenerator
        """
        print('Loading Test ImageDataGenerator'.center(80, '-'))

        pure_gen = fm.get_image_data_generator(augment=False)
        test_dir = os.path.join(TEMP_IMAGE_DIR, 'test')

        # refresh directories
        os.makedirs(test_dir, exist_ok=True)
        shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)
        test_collection.export_by_label(test_dir, balancing=1)

        print('[debug] test image count: {}'.format(
            test_collection.df.shape[0]))

        test_gen = pure_gen.flow_from_directory(
            test_dir,
            target_size=fm.get_output_shape(),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
        )
        print('Test images: {}'.format(test_gen.n))

        return test_gen

# In[4]:

    def optimize_learning_rate(fm: FineModel, depth_index, train_gens,
                               val_gens, test_gen):
        """
        Train the fine model (frozen at some given depth) for all five folds of data,
        and choose the optimal learning rate BASED ON THE FINAL VALIDATION ACCURACY.
        Consider learning rates defined in the global variable LEARNING_RATES
        
        
        Save model with the following KEYS: [load weights via fm.load_weights(KEY)]
        EXP01_D01
        Fully trained model for the optimal learning rate
        
        
        :param fm:
        FineModel to train, i.e., the base network to train on
        
        :param depth_index:
        The INDEX of the "freeze depth" for the given FineModel
        
        :param train_gens
        List of train ImageDataGenerators for each fold
        
        :param val_gens  
        List of validation ImageDataGenerators for each fold
        
        :param val_gens  
        Test ImageDataGenerator for each fold
        
        :return: None
        """

    def train_model_all_folds(fm, depth_index, lr_index, epochs, train_gens,
                              val_gens, test_gen):
        """
        Train the model (frozen at some depth) for all five folds


        Saves intermediate models with the following KEYS: [load weights via fm.load_weights(KEY)]
        EXP01_D01_L03_F01:
        Fully trained model for the 1st freeze depth, 3rd learning rate, fold 1
        EXP01_D01_L03_F01_E025:
        Partially trained model for the 1st freeze depth, 3rd learning rate, fold 1, until the 25th epoch

        Saves training history with the following KEYS: [get data via ch.get_history(model_name, KEY)]
        EXP01_D01_L03_F01:
        Training history for the 1st freeze depth, 3rd learning rate, fold 1


        :param fm:
        FineModel to train, i.e., the base network to train on

        :param depth_index:
        The INDEX of the "freeze depth" for the given FineModel

        :param lr_index:
        The INDEX of the learning rate, i.e., lr = LEARNING_RATES[lr_index]

        :param epochs:
        Number of epochs to train. MUST BE MULTIPLE OF 5.

        :param train_gens
        List of train ImageDataGenerators for each fold

        :param val_gens
        List of validation ImageDataGenerators for each fold

        :param val_gens
        Test ImageDataGenerator for each fold

        :return:
        tuple(val_loss, val_acc): AVERAGE validation loss and accuracy at FINAL EPOCH
        """
        _depth_key = 'EXP01_D{:02}'
        _fold_key = 'EXP01_D{:02}_L{:02}_F{:02}'
        _epoch_key = 'EXP01_D{:02}_L{:02}_F{:02}_E{:03}'

        lr = LEARNING_RATES[lr_index]
        loss_list = []
        acc_list = []

        # train the model K times, one for each fold
        for i in range(K):
            # load model at previous state
            previous_depth_index = depth_index - 1
            if previous_depth_index < 0:
                fm.reload_model()
            else:
                fm.load_weights(_depth_key.format(previous_depth_index))
            fm.set_depth(depth_index)
            fm.compile_model(lr=lr)
            model = fm.get_model()

            print('[debug] batch: {}'.format(BATCH_SIZE))
            print('[debug] size: {}'.format(train_gens[i].n))
            print('[debug] steps: {}'.format(len(train_gens[i])))

            # train 5 epochs at a time
            T = 5  # model save interval in epochs
            start_epoch = 0
            while start_epoch < epochs:
                print('[debug] epoch {}'.format(start_epoch))
                target_epoch = start_epoch + T
                if target_epoch > epochs:
                    target_epoch = epochs
                result = model.fit_generator(
                    train_gens[i],
                    validation_data=val_gens[i],
                    steps_per_epoch=len(train_gens[i]),
                    validation_steps=len(val_gens[i]),
                    workers=16,
                    use_multiprocessing=True,
                    shuffle=True,
                    epochs=target_epoch,
                    initial_epoch=start_epoch,
                )
                start_epoch = target_epoch

                # update training history
                ch.append_history(result.history, fm.get_name(),
                                  _fold_key.format(depth_index, lr_index, i))
                # save intermediate weights
                fm.save_weights(
                    _epoch_key.format(
                        depth_index,
                        lr_index,
                        i,
                        target_epoch,
                    ))

            # save final weights
            fm.save_weights(_fold_key.format(depth_index, lr_index, i))

            print('[debug] test size: {}'.format(test_gen.n))
            print('[debug] test steps: {}'.format(len(test_gen)))

            loss, acc = model.evaluate_generator(
                test_gen,
                steps=len(test_gen),
                workers=16,
                use_multiprocessing=True,
            )

            print('[debug] test_loss={}, test_acc={}'.format(loss, acc))

            loss_list.append(loss)
            acc_list.append(acc)

        print('Exporting analysis')
        for metric in analysis.metric_names.keys():
            analysis.analyze_lr(fm, fm.get_name(), depth_index, lr_index, lr,
                                metric)

        total_loss = 0
        for loss in loss_list:
            total_loss += loss
        avg_loss = total_loss / K

        total_acc = 0
        for acc in acc_list:
            total_acc += acc
        avg_acc = total_acc / K

        print('[debug] avg_test_loss={}, avg_test_acc={}'.format(
            avg_loss, avg_acc))

        return avg_loss, avg_acc

# ## Load Data

# ### Select train/test data and print statistics

# In[ ]:

    train = cri.CrCollection.load().filter_by(
        dataset_index=0).tri_label().labeled()
    test = cri.CrCollection.load().filter_by(
        dataset_index=1).tri_label().labeled()

    if SAMPLE:
        train = train.sample(frac=0.1)
        test = test.sample(frac=0.1)

    def print_stats(collection):
        df = collection.df
        print('{:<3} patients / {:<4} images'.format(df.pid.unique().shape[0],
                                                     df.shape[0]))
        print(df.label.value_counts().to_string())

    print('Training/Validation Set'.center(80, '-'))
    print_stats(train)

    print('Test Set'.center(80, '-'))
    print_stats(test)

    print()
    print(
        'Note that OAP, OBS images in the training/validation set will be duplicated 5 times'
    )
    print('to solve the class imbalance issue')
    print()

    # ### Print statistics on 5-fold split data

    # In[ ]:

    folds = train.k_split(K)

    stats = dict()
    for i, fold in enumerate(folds):
        counts = fold.df.label.value_counts()
        counts.loc['total'] = fold.df.shape[0]
        stats[i + 1] = counts
    stats = pd.DataFrame(stats)

    print('5-Fold Training Set Data'.center(80, '-'))
    print(stats.to_string(col_space=8))

    keys = [
        #'xception',
        #'mobileneta25',
        #'mobilenetv2a35',
        #'vgg16',
        'resnet50v2',
        #'inception_v3',
        #'inception_resnet_v2',
        #'densenet121',
        #'nasnet_mobile',
    ]

    models = FineModel.get_dict()

    for key in keys:
        keras.backend.clear_session()
        fm = models[key]()
        train_gens, val_gens = get_train_val_generators(fm, folds)
        test_gen = get_test_generator(fm, test)
        for i, lr in enumerate(LEARNING_RATES):
            print('Starting training {} lr={}'.format(fm.get_name(),
                                                      lr).center(100, '-'))
            train_model_all_folds(fm, 0, i, EPOCHS, train_gens, val_gens,
                                  test_gen)

except Exception as e:
    error = traceback.format_exc()
    error += '\n'
    error += str(e)
    print(error)
    notify(error)
