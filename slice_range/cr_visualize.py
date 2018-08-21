import os
import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt

import cr_interface as cri
import scipy.misc
import scipy.ndimage


#LABELS = [None, 'oap', 'ap', 'md', 'bs', 'obs']
LABELS = [None, 'oap', 'in', 'obs']

DISPLAY_NAME = {
    'oap': 'OUT_AP',
    'ap': 'AP',
    'md': 'MID',
    'bs': 'BS',
    'obs': 'OUT_BS',
    'nan': '-'
}

axes = []
predictions = None

current_label = None
fig = plt.figure()
index = 0
last_index = None


def main():
    global metadata, results, predictions, image_collection
    # image_collection: [((db_index: int, subject_index: int), [ cr_code: str ])]

    metadata = cri.load_metadata()
    for p in metadata:
        if 'label' in p:
            print(p['label'])
    result = cri.prompt_and_load_result()
    p = result['predictions']

    predictions = []
    truths = []
    for result in p.values():
        predictions.append(result['prediction'])
        truths.append(result['truth'])

    y_test = truths
    y_pred = predictions

    class_names = sorted(LABELS[1:])
    ############################
    ############################
    ############################
    ############################
    class_names = sorted(['oap', 'oba', 'in'])

    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()



if __name__ == "__main__":
    main()
