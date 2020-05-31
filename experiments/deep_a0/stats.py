import pandas as pd
import cr_interface as cri
from cr_interface import CrCollection

from params import *


def print_collection_stats(collection: CrCollection, title='Dataset'):
    df = collection.df
    print('{} Stats'.format(title).center(80, '-'))
    print('{:<3} patients / {:<4} images'.format(df.pid.unique().shape[0],
                                                 df.shape[0]))
    print(df.label.value_counts().to_string())


def print_fold_stats(folds):
    """
    - folds: List of CrCollections returned from CrCollection.k_split()
    """
    print()
    print('Note that OAP, OBS images in the training/validation set will be')
    print('duplicated 5 times to solve the class imbalance issue')
    print()

    # Print number of images by fold by label (training data)
    stats = dict()
    for i, fold in enumerate(folds):
        counts = fold.df.label.value_counts()
        counts.loc['total'] = fold.df.shape[0]
        stats[i + 1] = counts
    stats = pd.DataFrame(stats)
    print('{}-Fold Training Set'.format(K).center(80, '-'))
    print(stats.to_string(col_space=8))
    print()

    # Columnwise-print or cr_codes (training data)
    cr_codes_by_fold = list(sorted(fold.df.pid.unique()) for fold in folds)
    max_len = 0
    for codes in cr_codes_by_fold:
        if max_len < len(codes):
            max_len = len(codes)
    for i, _ in enumerate(folds):
        print('Fold {}'.format(i + 1).ljust(16), end='')
    print()
    print('-' * 80)
    for i in range(max_len):
        for codes in cr_codes_by_fold:
            if i < len(codes):
                print('{:<16d}'.format(codes[i]), end='')
            else:
                print('{:<16s}'.format(''), end='')
        print()
    print()


if __name__ == '__main__':
    train = cri.CrCollection.load().filter_by(
        dataset_index=0).tri_label().labeled()
    print_collection_stats(train, title="Train Set")
    test = cri.CrCollection.load().filter_by(
        dataset_index=1).tri_label().labeled()
    print_collection_stats(test, title="Test Set")
