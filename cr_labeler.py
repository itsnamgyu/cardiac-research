import argparse
import collections
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import cr_interface as cri
from core import cam, paths
from core.fine_model import FineModel
from core.result import Result

LABELS = [None, 'oap', 'in', 'obs']
# LABELS = [None, 'oap', 'ap', 'md', 'bs', 'obs']

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

show_predictions = False
predictions = None
percentages = None  # { 'oap': '0.009' ... } Note: it's a string number

show_cam = False  # class activation maps
cam_fm: FineModel = None  # model used for CAM

show_chart = False

all_bars = []

current_label = None
fig = plt.figure(figsize=(8, 5))
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
    title = title.format(patient[0][0], patient[0][1], index + 1,
                         len(image_collection), label)

    return title


def update_plot():
    global metadata, results, predictions, percentages, image_collection
    global index, last_index, current_label, show_chart, all_bars, all_texts, show_predictions

    patient = image_collection[index]

    if last_index != index:
        all_bars = []
        all_texts = []
        del axes[:]
        fig.clf()

        for i, cr_code in enumerate(patient[1]):
            path = os.path.join(cri.DATABASE_DIR, cr_code + '.jpg')

            axes.append(fig.add_subplot(4, 6, i + 1))
            extent = (0, 10, 0, 10)

            image = plt.imread(path)

            # Class activation maps
            if show_cam:
                # Convert to rgb image
                image = np.stack((image, ) * 3, axis=-1)
                # Resize
                image = np.array(
                    Image.fromarray(image).resize(cam_fm.get_output_shape()))
                # Apply gradcam
                image = cam.overlay_gradcam(cam_fm, image)
                axes[i].imshow(image, extent=extent)
            else:
                # Display grayscale image
                axes[i].imshow(image, cmap='gray', extent=extent)

            if show_predictions:
                if cr_code not in predictions or cr_code not in percentages:
                    warnings.warn('{} not in predictions'.format(cr_code))
                else:
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
                        text = axes[i].text(x_locations[j],
                                            p * 8 + 0.5,
                                            '%d' % (p * 100),
                                            color=(1, 1, 0),
                                            horizontalalignment='center',
                                            bbox=dict(facecolor='black',
                                                      alpha=0.5))
                        text.set_fontsize(8)
                        all_texts.append(text)

                    if truth:
                        bars[LABELS[1:].index(truth)].set_color(right_color)

            axes[i].set_axis_off()

    for i, cr_code in enumerate(patient[1]):
        # hotfix: convert to tri-label
        truth = metadata[cr_code].get('label', 'nan')
        if 'in' in LABELS:
            if truth and truth in 'apmdbs':
                truth = 'in'
        truth = DISPLAY_NAME[truth]
        origin = DISPLAY_NAME[metadata[cr_code].get('label', 'nan')]
        if show_predictions:
            if cr_code not in predictions or cr_code not in percentages:
                warnings.warn('{} not in predictions'.format(cr_code))
            else:
                prediction = DISPLAY_NAME[predictions[cr_code]]

                label = 'T={} / P={}'.format(origin, prediction)
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
    global metadata, results, predictions, image_collection, index, last_index, current_label, show_chart, show_predictions
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
    except (ValueError, KeyError, IndexError):
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
    global metadata, results, predictions, percentages, image_collection, LABELS, show_cam, cam_fm, show_predictions, index

    metadata = cri.load_metadata()
    for p in metadata:
        if 'label' in p:
            print(p['label'])

    parser = argparse.ArgumentParser()
    description = 'Start in prediction mode. Note that in predicitons mode,' \
        'you can press the spacebar to use the predictions to label the images'
    parser.add_argument('-P',
                        '--predictions',
                        help=description,
                        action='store_true')
    description = 'Show class activation maps in prediction mode'
    parser.add_argument('-C', '--cam', help=description, action='store_true')
    description = 'Export all plots'
    parser.add_argument('-E',
                        '--export',
                        help=description,
                        action='store_true')
    args = parser.parse_args()

    show_cam = args.cam
    show_predictions = args.predictions or args.cam

    if show_predictions:
        if args.cam:

            def _output_filter(e, m, i):
                result = paths.get_test_result_path(e, m, i)
                weights = paths.get_weights_path(e, m, i)
                return os.path.exists(result) and os.path.exists(weights)
        else:

            def _output_filter(e, m, i):
                result = paths.get_test_result_path(e, m, i)
                return os.path.exists(result)

    if show_predictions:
        output_key = paths.select_output(_output_filter)
        if not output_key:
            return None
        e, m, i = output_key
        result = Result.load(exp_key=e, model_key=m, instance_key=i)
        result_dict = result.data

        p = result_dict['predictions']
        import json
        print('Predictions: {}'.format(json.dumps(p, indent=4)))

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

    if show_cam:
        try:
            print('Loading {} for CAM analysis'.format(output_key))
            fm = FineModel.load_by_key(m)
            fm.load_weights(exp_key=e, instance_key=i)
        except Exception:
            raise RuntimeError('Failed to load corresponding model weights')
        cam_fm = fm

    image_collection = sorted(image_collection.items())

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_button_press)

    plt.subplots_adjust(top=0.95,
                        bottom=0.05,
                        right=1,
                        left=0,
                        hspace=0.2,
                        wspace=0)

    if args.export:
        export_dir = os.path.abspath('labeler_exports')
        os.makedirs(export_dir, exist_ok=True)
        print('Exporting all images to {}'.format(export_dir))
        for i in tqdm(range(len(image_collection))):
            index = i
            update()
            patient = image_collection[i]
            basename = '[{:03d}] D{:02d}_P{:08d}.png'.format(
                i, patient[0][0], patient[0][1])
            path = os.path.join(export_dir, basename)
            plt.savefig(path,
                        dpi=320,
                        transparent=False,
                        bbox_inches=None,
                        pad_inches=0.1)
    else:
        update()
        plt.show()

    cri.save_metadata(metadata)


if __name__ == "__main__":
    main()
