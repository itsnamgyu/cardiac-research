import os
import argparse
import collections

import numpy as np
import matplotlib.pyplot as plt

import cr_interface as cri
import scipy.misc
import scipy.ndimage
from sklearn import linear_model


LABELS = [None, 'oap', 'ap', 'md', 'bs', 'obs']

DISPLAY_NAME = {
    'oap': 'OUT_AP',
    'ap': 'AP',
    'md': 'MID',
    'bs': 'BS',
    'obs': 'OUT_BS',
    'nan': '-',
    'in': 'IN',
}

axes = []
predictions = None
percentages = None  # { 'oap': '0.009' ... } Note: it's a string number

show_chart = False

all_bars = []

current_label = None
fig = plt.figure()
index = 0
last_index = None


'''
Global Variables

- image_collection: [
    (db_index: int, subject_index: int, phase_index: int), [cr_code: str, ...]),
    ...
]

'''

def get_window_title():
    global metadata, results, predictions, percentages, image_collection, index, current_label
    patient = image_collection[index]

    if current_label:
        label = DISPLAY_NAME[current_label]
    else:
        label = 'NO_LABEL'

    title = 'DB#{:02d} PATIENT{:08d} ({:03d}/{:03d}) [ {:^10s} ]'
    title = title.format(patient[0][0], patient[0]
                         [1], index + 1, len(image_collection), label)

    return title


def update_plot():
    global metadata, results, predictions, percentages, image_collection
    global index, last_index, current_label, show_chart, all_bars, all_texts


    def regress(values):
        # values: [ 0, 2, 1, 3 ]
        # output: [ 0, 1, 2, 3 ]
        regr = linear_model.LinearRegression()
        regr.fit(np.arange(len(values)).reshape(-1 ,1), values)
        return regr.predict(np.arange(len(values)).reshape(-1, 1))

    patient = image_collection[index]

    if last_index != index:
        all_bars = []
        all_texts = []
        del axes[:]
        fig.clf()

        for i, cr_code in enumerate(patient[1]):
            path = os.path.join(cri.DATABASE_DIR, cr_code + '.jpg')

            axes.append(fig.add_subplot(4, 6, i + 1))

            image = scipy.ndimage.imread(path)
            extent = (0, 10, 0, 10)
            axes[i].imshow(image, cmap=plt.cm.gray, extent=extent)

            if percentages:
                patient_percentages = []
                for label in LABELS[1:]:
                    patient_percentages.append(
                        float(percentages[cr_code][label]))

                # hotfix: convert to tri-label
                truth = metadata[cr_code].get('label', None)
                if 'in' in LABELS:
                    if truth and truth in 'apmdbs':
                        truth = 'in'

                if not truth or predictions[cr_code] == truth:
                    wrong_color = (0.75, 0.75, 0.75)
                    right_color = (0.75, 0.75, 0.75)
                else:
                    wrong_color = (1, 0, 0)
                    right_color = (0, 1, 0)

                x_locations = np.linspace(1, 9, len(patient_percentages))
                bars = axes[i].bar(x_locations,
                                   np.array(patient_percentages) * 8,
                                   color=wrong_color)
                all_bars.extend(bars)
                for j, p in enumerate(patient_percentages):
                    text = axes[i].text(x_locations[j], p * 8 + 0.5, '%d' % (p * 100),
                                        color=(1, 1, 0), horizontalalignment='center',
                                        bbox=dict(facecolor='black', alpha=0.5))
                    text.set_fontsize(8)
                    all_texts.append(text)

                if truth:
                    bars[LABELS[1:].index(truth)].set_color(right_color)

            axes[i].set_axis_off()

    if percentages:
        weighted_averages = []
        for cr_code in patient[1]:
            avg = 0
            for i, label in enumerate(LABELS[1:]):
                avg += float(percentages[cr_code][label]) * (i + 1)
            weighted_averages.append(avg)
        regressed_averages = regress(weighted_averages)

    for i, cr_code in enumerate(patient[1]):
        # hotfix: convert to tri-label
        truth = metadata[cr_code].get('label', 'nan')
        if 'in' in LABELS:
            if truth and truth in 'apmdbs':
                truth = 'in'
        truth = DISPLAY_NAME[truth]
        origin = DISPLAY_NAME[metadata[cr_code].get('label', 'nan')]
        if predictions:
            prediction = DISPLAY_NAME[predictions[cr_code]]

            label = '{} / {} (P)'.format(origin, prediction)
            #label += ' [{:.2f}]'.format(regressed_averages[i])
            if truth == '-':
                color = (0.2, 0.2, 0.2)
            else:
                if truth == prediction:
                    color = (0, 0.6, 0)
                else:
                    color = (0.75, 0, 0)
        else:
            color = (0, 0, 0)
            label = truth
        axes[i].set_title(label, fontsize='small', snap=True, color=color)

    for bar in all_bars:
        if show_chart:
            bar.set_alpha(0.5)
        else:
            bar.set_alpha(0)
    for text in all_texts:
        if show_chart:
            text.set_alpha(1)
            text.get_bbox_patch().set_alpha(0.5)
        else:
            text.set_alpha(0)
            text.get_bbox_patch().set_alpha(0)

    fig.canvas.set_window_title(get_window_title())
    fig.canvas.draw()


def update():
    update_plot()


def on_key_press(event):
    global metadata, results, predictions, image_collection, index, last_index, current_label, show_chart
    last_index = index

    if event.key == 'left':
        index -= 1
    if event.key == 'right':
        index += 1

    if event.key == 'h':
        index -= 10
    if event.key == 'l':
        index += 10

    if event.key == 'c':
        show_chart = not show_chart

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
    global metadata, results, image_collection, index, last_index
    patient = image_collection[index]

    for i, ax in enumerate(axes):
        if event.inaxes == ax:
            cr_code = patient[1][i]
            if current_label:
                metadata[cr_code]['label'] = current_label
            else:
                del metadata[cr_code]['label']

    fig.canvas.draw()
    update()


def main():
    global metadata, results, predictions, percentages, image_collection, LABELS

    metadata = cri.load_metadata()
    for p in metadata:
        if 'label' in p:
            print(p['label'])

    parser = argparse.ArgumentParser()
    description = \
        '''Start in prediction mode. Note that in predicitons mode,
    you can press the spacebar to use the predictions to label the images'''
    parser.add_argument('-P', '--predictions', help=description,
                        action='store_true')
    args = parser.parse_args()

    if args.predictions:
        result_dict = cri.prompt_and_load_result()
        p = result_dict['predictions']

        # hotfix
        if cri.is_tri_label_result(result_dict):
            LABELS = [None, 'oap', 'in', 'obs']

        predictions = {}
        percentages = {}
        for basename, result in p.items():
            cr_code = cri.extract_cr_code(basename)
            predictions[cr_code] = result['prediction']
            percentages[cr_code] = result['percentages']

        image_collection = {}
        for basename, result in predictions.items():
            cr = cri.parse_cr_code(basename, match=False)
            image_collection[tuple(cr[:3])] = []

        # get list of patients then add all of their images (not just from predictions)
        for cr_code in metadata.keys():
            cr = cri.parse_cr_code(cr_code)
            if tuple(cr[:3]) in image_collection:
                image_collection[tuple(cr[:3])].append(cr_code)
    else:
        image_collection = collections.defaultdict(list)
        for cr_code in metadata.keys():
            cr = cri.parse_cr_code(cr_code)
            image_collection[tuple(cr[:3])].append(cr_code)

    image_collection = sorted(image_collection.items())

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_button_press)

    plt.subplots_adjust(top=0.95, bottom=0.05, right=1, left=0,
                        hspace=0.2, wspace=0)
    update()
    plt.show()

    cri.save_metadata(metadata)


if __name__ == "__main__":
    main()
