from collections import defaultdict
import math
from bayes_opt import BayesianOptimization
import keras_utils as ku

import core.history as ch
import core.fine_model as cm
from core.fine_model import FineModel


BATCH_SIZE = 32
T = 10

def train_k_by_t_epochs(fm, lr, decay, train_gen, val_gen, key, t=T):
    fm.load_weights(key, verbose=0)
    fm.compile_model(lr=lr, decay=decay)
    result = fm.get_model().fit_generator(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=math.ceil(train_gen.n / BATCH_SIZE),
        validation_steps=len(val_gen),
        shuffle=True,
        epochs=t)
    ch.append_history(result.history, fm.get_name(), key)
    fm.save_weights(key, verbose=0)


def train_k(train_collection, fm, lr, decay, k=5):
    '''
    Returns (loss, epochs)
    '''
    print('Converging {} [lr={}, decay={}]'.format(fm.get_name(), lr, decay).center(100, '-'))
    
    k_collections = train_collection.k_split(k)
    _key = 'K{:02d}'
    _train_dir = 'temp_images/train_' + _key
    _val_dir = 'temp_images/val_' + _key
    
    aug_gen = fm.get_image_data_generator(augment=True)
    pure_gen = fm.get_image_data_generator(augment=False)
    
    train_gens = []
    val_gens = []
    
    print('Exporting images... ', end='')
    for i, collection in enumerate(k_collections):
        for j in range(5):
            train_dir = _train_dir.format(j)
            val_dir = _val_dir.format(j)
            if i == j:
                collection.export_by_label(val_dir)
            else:
                collection.export_by_label(train_dir)
    print('complete.', end='')
                
    for i in range(k):
        train_gens.append(aug_gen.flow_from_directory(
            train_dir,
            target_size=fm.get_output_shape(),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        ))

        val_gens.append(aug_gen.flow_from_directory(
            val_dir,
            target_size=fm.get_output_shape(),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        ))
                         
    for i in range(k):
        fm.save_weights(_key.format(i), verbose=0)
        
    while True:
        print('Training {} epochs for k splits... '.format(T), end='')
        histories = []
        for i in range(k):
            key = _key.format(i)
            train_k_by_t_epochs(fm, lr, decay, train_gens[i], val_gens[i], key)
            histories.append(ch.load_history(fm.get_name(), key))
        history = ch.get_average(histories)
        index, value = ch.get_early_stop_index_and_value(history)
        if index is not None:
            print('converged!.')
            print('Validation Loss: {}, Epoch {}'.format(value, index))
            return history.loc[index, 'val_loss'], index
        print('converging...')


def optimize_hyperparameters(train_collection, fm, lr=(1e-4, 1e-1), decay=(0, 1e-3), init_points=5, n_iter=5):
    '''
    lr, decay: (lower_bound, upper_bound)
    Returns (lr, decay, epochs)
    '''
    print('Optimizing {}'.format(fm.get_name()))
    
    results = defaultdict(list)
    def f(lr, decay):
        loss, epochs = train_k(train_collection, fm, lr, decay)
        results['lr'].append(lr)
        results['decay'].append(decay)
        results['loss'].append(loss)
        results['epochs'].append(epochs)
        return -loss
    
    pbounds = {
        'lr': lr,
        'decay': decay
    }
    
    optimizer = BayesianOptimization(f=f, pbounds=pbounds, random_state=1)
    optimizer.maximize(init_points, n_iter)
    
    loss = -optimizer.max['target']
    lr = optimizer.max['params']['lr']
    decay = optimizer.max['params']['decay']
    
    print('Validation Loss of {} for [lr={}, decay={}]'.format(loss, lr, decay))
    
    for i in range(len(results['lr'])):
        if results['decay'][i] == decay and results['lr'][i] == lr:
            epochs = results['epochs'][i]
            
    return lr, decay, epochs


def optimize_full_model(train_collection, test_collection, fm):
    print('FULLY TRAINING {}'.format(fm.get_name().upper()).center(100, '='))
    
    aug_gen = fm.get_image_data_generator(augment=True)
    pure_gen = fm.get_image_data_generator(augment=False)
    
    train_dir = 'temp_images/train'
    test_dir = 'temp_images/test'
    train_collection.export_by_label(train_dir)
    test_collection.export_by_label(test_dir)
    train_gen = aug_gen.flow_from_directory(
        train_dir,
        target_size=fm.get_output_shape(),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_gen = pure_gen.flow_from_directory(
        test_dir,
        target_size=fm.get_output_shape(),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    fm.reload_model()
    
    for d in range(len(fm.depths)):
        print('TRAINING DEPTH #{}'.format(d).center(100, '='))
        key = 'D{:02d}'.format(d)
        fm.save_weights(key, verbose=0)
        fm.set_depth(d)
        
        lr, decay, epochs = optimize_hyperparameters(train_collection, fm)
        with open('hp_{}_d{}'.format(fm.get_name(), d)) as f:
            f.write('lr,decay,epochs\n')
            f.write('{},{},{}\n'.format(lr, decay, epochs))
        
        fm.load_weights(key, verbose=0)
        result = fm.get_model().fit_generator(
            train_gen,
            steps_per_epoch=math.ceil(train_gen.n / BATCH_SIZE),
            shuffle=True,
            epochs=t)
        
        evaluation = fm.get_model().evaulate_generator(test_gen)
        fm.save_weights(key, verbose=0)
        
        print('DEPTH #{} TRAIN RESULTS'.format(d))
        print(evaluation)


def optimize_all_models(train_collection, test_collection):
    for fm_class in FineModel.get_list():
        optimize_full_model(train_collection, test_collection, fm_class())
