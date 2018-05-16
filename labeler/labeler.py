import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict
from dicom_dataset import DicomDataset
import scipy.misc
from enum import Enum

DISPLAY_NAMES = [
        'NaN', 'OUT_AP', 'AP',
        'MID', 'BA', 'OUT_BA',
        ]


class Label(Enum):
    UNLABELED = 0
    OUT_OF_APICAL = 1
    APICAL = 2
    MIDDLE = 3
    BASAL = 4
    OUT_OF_BASAL = 5

    def display_name(self):
        return DISPLAY_NAMES[self.value]


dcm_dataset: DicomDataset = None
dcm_labels: Dict[DicomDataset.NamedDicom, Label] = {}
subjects: List[DicomDataset.Subject] = []

subject_index = 0
displayed_subject: DicomDataset.Subject = None

axes = []

current_label = Label.UNLABELED

load_error = False
fig = plt.figure()


SUBJECT_DIRECTORY_FORMAT = "DET{:07d}"
LABEL_FILE = 'label_data.npy'


def get_window_title():
    global load_error
    global dcm_labels, subjects, subject_index, current_subject, displayed_subject

    subject = subjects[subject_index]
    if load_error:
        title = 'Error Loading Subject #{} [{}]'.format(subject_index, subject)
    else:
        title = 'Subject #{} [{}]'.format(subject_index, subject.id)
    title += ' [{}]'.format(current_label.display_name())

    return title


def update_plot():
    global dcm_labels, subjects, subject_index, displayed_subject

    fig.canvas.set_window_title(get_window_title())

    subject = subjects[subject_index]
    if displayed_subject is not subject:
        displayed_subject = subject

        fig.clf()
        del axes[:]
        for i, ndcm in enumerate(subject.get_sorted_ndcms()):
            axes.append(fig.add_subplot(4, 6, i + 1))
            ndcm.load()
            axes[i].imshow(ndcm.img_processed, cmap=plt.cm.gray)
            axes[i].set_axis_off()


    for i, ndcm in enumerate(subject.get_sorted_ndcms()):
        label = dcm_labels.get(ndcm, Label.UNLABELED).display_name()
        axes[i].set_title('{} [{:.4f}]'.format(label, ndcm.get_data().SliceLocation), fontsize='small', snap=True)

    fig.canvas.draw()


def update():
    update_plot()


def on_key_press(event):
    global dcm_labels, subjects, subject_index, displayed_subject, current_label

    if event.key == 'left':
        subject_index -= 1
    if event.key == 'right':
        subject_index += 1

    if event.key == 'h':
        subject_index -= 10
    if event.key == 'l':
        subject_index += 10

    subject_index %= len(subjects)

    try:
        current_label = Label(int(event.key))
    except(ValueError, KeyError):
        pass

    update()


def on_button_press(event):
    global subject_index, axes, subjects, dcm_labels, dicom_dict, current_label

    fig.canvas.draw()

    subject = subjects[subject_index]
    for i, ax in enumerate(axes):
        if event.inaxes == ax:
            ndcm = subject.get_sorted_ndcms()[i]
            dcm_labels[ndcm] = current_label

    update()


def load_dcm_labels(dcm_dataset: DicomDataset, label_file=LABEL_FILE):
    dcm_name_labels: Dict[str, Label] = {}

    try:
        ndarray = np.load(label_file)
        for i in range(ndarray.shape[0]):
            dcm_name_labels[ndarray[i][0]] = int(ndarray[i][1])
    except OSError:
        print("could not find data file: {}".format(LABEL_FILE))
        print("initializing new data")
        return {}

    dcm_labels: Dict[DicomDataset.NamedDicom, Label] = {}
    for subject in dcm_dataset.subjects.values():
        for ndcm in subject.named_dcms:
            if ndcm.name in dcm_name_labels:
                try:
                    dcm_labels[ndcm] = Label(dcm_name_labels[ndcm.name])
                except ValueError:
                    print('issue with label data')
                    print('initializing new data')
                    return {}

    print('successfully loaded label data from {}'.format(label_file))

    return dcm_labels


def save_dcm_labels(dcm_labels: Dict[DicomDataset.NamedDicom, Label], label_file=LABEL_FILE):
    array = []
    for ndcm, label in dcm_labels.items():
        array.append((ndcm.name, label.value))
    np.save(label_file, np.array(array))
    print('data successfully saved to {}'.format(label_file))


def main():
    global dcm_dataset
    global subject_index, axes, subjects, dcm_labels, current_label
    global dicom_dict, dcm_labels, subjects, LABEL_FILE

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', dest='dir',  type=str, nargs=1, default='cap_challenge',
                        help='top directory of dicom files')

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print('could not find data directory {}'.format(args.dir))
        print('terminating...')
        exit()

    dcm_dataset = DicomDataset(data_directory=args.dir, preload_data=True)
    subjects = list(dcm_dataset.subjects.values())

    dcm_labels = load_dcm_labels(dcm_dataset)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_button_press)

    update()
    plt.show()

    save_dcm_labels(dcm_labels)

    # save as jpegs
    print('saving labeled images...')

    try:
        shutil.rmtree('cap_labeled')
    except FileNotFoundError:
        pass

    for ndcm, label in dcm_labels.items():
        directory = os.path.join('cap_labeled', label.name)
        jpg_path = os.path.join(directory, ndcm.name + '.jpg')

        try:
            os.makedirs(directory)
        except OSError:
            pass

        image_array = ndcm.get_data().pixel_array
        scipy.misc.imsave(jpg_path, image_array)

    print('labeled images saved')



if __name__ == "__main__":
    main()
