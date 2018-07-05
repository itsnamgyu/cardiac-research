import os
import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt

import cr_interface as cri
import scipy.misc
import scipy.ndimage


LABELS = [None, 'oap', 'ap', 'md', 'bs', 'obs']

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


def get_window_title():
    global metadata, results, predictions, image_collection, index, current_label
    patient = image_collection[index]

    if current_label:
        label = DISPLAY_NAME[current_label]
    else:
        label = 'NO_LABEL'


    title = 'DB#{:02d} PATIENT{:08d} ({:03d}/{:03d}) [ {:^10s} ]'
    title = title.format(patient[0][0], patient[0][1], index + 1, len(image_collection), label)

    return title


def update_plot():
    global metadata, results, predictions, image_collection, index, last_index, current_label
    patient = image_collection[index]

    if last_index != index:
        del axes[:]
        fig.clf()

        for i, cr_code in enumerate(patient[1]):
            path = os.path.join(cri.DATABASE_DIR, cr_code + '.jpg')

            axes.append(fig.add_subplot(4, 6, i + 1))

            image = scipy.ndimage.imread(path)
            axes[i].imshow(image, cmap=plt.cm.gray)
            axes[i].set_axis_off()
    
    for i, cr_code in enumerate(patient[1]):
        l = DISPLAY_NAME[metadata[cr_code].get('label', 'nan')]
        label = '{}'.format(DISPLAY_NAME[metadata[cr_code].get('label', 'nan')])
        if predictions:
            p = DISPLAY_NAME[predictions[cr_code]]
            label += ' / {} (P)'.format(DISPLAY_NAME[predictions[cr_code]])
            if l == '-':
                color = (0.2, 0.2, 0.2)
            else:
                if l == p:
                    color = (0, 0.6, 0)
                else:
                    color = (0.75, 0, 0)
        else:
            color = (0, 0, 0)
        axes[i].set_title(label, fontsize='small', snap=True, color=color)

    fig.canvas.set_window_title(get_window_title())
    fig.canvas.draw()


def update():
    update_plot()


def on_key_press(event):
    global metadata, results, predictions, image_collection, index, last_index, current_label
    last_index = index

    if event.key == 'left':
        index -= 1
    if event.key == 'right':
        index += 1

    if event.key == 'h':
        index -= 10
    if event.key == 'l':
        index += 10

    if event.key == ' ':
        patient = image_collection[index]
        for i, cr_code in enumerate(patient[1]):
            metadata[cr_code]['label'] = predictions[cr_code]

    index %= len(image_collection)

    try:
        current_label = LABELS[int(event.key)]
    except(ValueError, KeyError, IndexError):
        pass

    update()


def on_button_press(event):
    global metadata, results, predictions, image_collection, index, last_index
    patient = image_collection[index]

    for i, ax in enumerate(axes):
        if event.inaxes == ax:
            cr_code = patient[i]
            if current_label:
                metadata[cr_code]['label'] = current_label
            else:
                del metadata[cr_code]['label']

    fig.canvas.draw()
    update()


def main():
    global metadata, results, predictions, image_collection
    # image_collection: [((db_index: int, subject_index: int), [ cr_code: str ])]

    metadata = cri.load_metadata()
    for p in metadata:
        if 'label' in p:
            print(p['label'])
    results = cri.load_results()

    parser = argparse.ArgumentParser()
    description = \
        '''Start in prediction mode. Note that in predicitons mode, 
    you can press the spacebar to use the predictions to label the images'''
    parser.add_argument('-P', '--predictions', help=description,
                        action='store_true')
    args = parser.parse_args()

    if args.predictions:
        print()
        print('{:-^80}'.format(' Predictions List '))
        for i, result in enumerate(results):
            print('%d.\tModule: %s' % (i, result['tfhub_module']))
            print('\tSteps: %-10sRate: %-10sAccuracy: %-10s' % (
                result['training_steps'],
                result['learning_rate'],
                result['test_accuracy'])
            )
            print()

        while True:
            try:
                index = int(input('Which of the predictions would you like to use? '))
                p = results[index]['predictions']
                break
            except (IndexError, ValueError):
                print('Invalid index')
                continue

        predictions = {}
        for basename, result in p.items():
            cr_code = cri.extract_cr_code(basename)
            predictions[cr_code] = result[0]

        image_collection = {}
        for basename, result in predictions.items():
            cr = cri.parse_cr_code(basename, match=False)
            image_collection[(cr[0], cr[1])] = []

        # get list of patients then add all of their images (not just from predictions)
        for cr_code in metadata.keys():
            cr = cri.parse_cr_code(cr_code)
            if (cr[0], cr[1]) in image_collection:
                image_collection[(cr[0], cr[1])].append(cr_code)
    else:
        image_collection = collections.defaultdict(list)
        for cr_code in metadata.keys():
            cr = cri.parse_cr_code(cr_code)
            image_collection[(cr[0], cr[1])].append(cr_code)

    image_collection = sorted(image_collection.items())

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_button_press)

    plt.subplots_adjust(top=0.95, bottom=0.05, right=1, left=0,
                        hspace=0.2, wspace=0)
    update()
    plt.show()


if __name__ == "__main__":
    main()
