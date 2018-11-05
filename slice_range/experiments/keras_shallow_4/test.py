import sys
sys.path.append('../..')

import math
import gc

import lib

try:
    import keras
    from keras import optimizers
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Dropout
    import sklearn
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    import cr_interface as cri
    import keras_utils as ku
    import keras_bottle as kb
    import keras_history as kh
    import keras_weights as kw

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

    TEST = True
    VERBOSE = 1
    LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    LEARNING_RATES = list(np.logspace(-2, -5, num=7, endpoint=True))


    def optimize(app, test=TEST, verbose=VERBOSE, batch_size=32):
        train_collection = cri.CrCollection.load().filter_by(
            dataset_index=0).tri_label().labeled()
        test_collection = cri.CrCollection.load().filter_by(
            dataset_index=1).tri_label().labeled()

        if test:
            epochs = 5
            n_aug = 1
            train_collection = train_collection.sample(frac=0.1)
            train_index = 999
            k = 5
            learning_rates = LEARNING_RATES
        else:
            epochs = 100
            n_aug = 5
            train_index = 0
            k = 5
            learning_rates = LEARNING_RATES

        k_train_sets, k_validation_sets, entire_set = kb.get_k_bottlenecks(
            app, train_collection, n_aug=n_aug, k=k, include_entire_set=True)

        min_loss_by_lr = []
        min_loss_epoch_by_lr = []
        average_histories = []

        if verbose >= 1:
            print('{}-fold, {}-learning-rate grid search'.format(
                k, len(learning_rates)).center(100, '-'))
        with tqdm(total=len(learning_rates) * k, disable=(verbose < 1)) as bar:
            for lr_index, lr in enumerate(learning_rates):
                histories = []
                for i in range(k):
                    train_bottles, train_labels = k_train_sets[i]
                    validation_bottles, validation_labels = k_validation_sets[i]

                    top_model = app.load_top_model(lr=lr)
                    total = len(train_bottles)
                    images_per_epoch = total / n_aug
                    batches_per_epoch = math.ceil(images_per_epoch / batch_size)

                    res = top_model.fit_generator(
                        bottleneck_generator(
                            train_bottles, train_labels, n_aug=n_aug, shuffle=True),
                        validation_data=bottleneck_generator(validation_bottles, validation_labels,
                                                             batch_size=len(validation_bottles), n_aug=1, shuffle=False),
                        steps_per_epoch=batches_per_epoch,
                        validation_steps=1,
                        epochs=epochs,
                        verbose=0)

                    kh.save_history(res.history, app.get_model(),
                                    lr_index, epochs, i)
                    histories.append(pd.DataFrame(res.history))
                    del top_model
                    gc.collect()
                    bar.update()
                keras.backend.clear_session()
                app.free_model()

                average_history = pd.concat(histories).groupby(level=0).mean()
                average_histories.append(average_history)
                min_loss_by_lr.append(average_history.val_loss.min())
                min_loss_epoch_by_lr.append(average_history.val_loss.idxmin())


        min_loss = pd.Series(min_loss_by_lr).min()
        m_lr_index = pd.Series(min_loss_by_lr).idxmin()
        m_acc = pd.Series(average_histories[m_lr_index].val_acc).min()
        m_epochs = min_loss_epoch_by_lr[m_lr_index]
        m_lr = learning_rates[m_lr_index]

        if verbose >= 1:
            print('[min_val_loss: {}] val_acc: {}, lr: {}, epochs: {}'.format(
                min_loss, m_acc, m_lr, m_epochs))

        if verbose >= 1:
            print('Training and testing final model'.center(100, '-'))

        train_bottles, train_labels = entire_set
        test_bottles, test_labels = kb.load_bottlenecks(
            app, test_collection, aug=False)
        test_labels = lib.onehot(test_labels)

        top_model = app.load_top_model(lr=m_lr)
        res = top_model.fit_generator(
            bottleneck_generator(train_bottles, train_labels,
                                 n_aug=n_aug, batch_size=batch_size, shuffle=True),
            steps_per_epoch=batches_per_epoch,
            epochs=m_epochs,
            verbose=0)
        kw.save_weights(top_model, train_index=train_index)
        kh.save_history(res.history, app.get_model(),
                        m_lr_index, m_epochs, split_index=None)

        loss, acc = top_model.evaluate(test_bottles, test_labels, verbose=0)
        if verbose >= 1:
            print('final accuracy: {}'.format(acc))
        kh.save_test_result(app.get_model(), loss, acc, m_lr, m_epochs)
        del top_model
        gc.collect()

        if verbose >= 1:
            print('Freeing bottlenecks and memory'.center(100, '-'))
        kb.reset_bottlenecks()
        keras.backend.clear_session()
        app.free_model()

        if verbose >= 1:
            print()
            print()
    ku.run_for_all_apps(optimize, title="5-fold 5-lr grid search optimization")
    lib.notify('CR Experiment 4 Test Successful')

except Exception as e:
    lib.notify('Issue CR Experiment 4 Test:\n{}'.format(str(e)))
    print(e)
