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


axes = []

current_label = Label.UNLABELED
fig = plt.figure()


def get_window_title():
    global load_error
    global dcm_labels, subjects, subject_index, current_subject, displayed_subject
    global color_labels, color_index

    subject = subjects[subject_index]
    if load_error:
        title = 'Error Loading Subject #{} [{}]'.format(subject_index, subject)
    else:
        title = 'Subject #{} [{}]'.format(subject_index, subject.id)
    title += ' [{}]'.format(current_label.display_name())
    title += ' [{}]'.format(color_labels[color_index])

    if not subject.sorted:
        title += ' [UNSORTED]'

    return title


def update_plot():
    global dcm_labels, subjects, subject_index, displayed_subject
    global color_labels, color_index, displayed_color

    subject = subjects[subject_index]
    if displayed_subject is not subject or displayed_color is not color_index:
        displayed_color = color_index
        displayed_subject = subject

        fig.clf()
        del axes[:]
        for i, ndcm in enumerate(subject.get_ndcms()):
            axes.append(fig.add_subplot(4, 6, i + 1))
            ndcm.load()
            axes[i].imshow(ndcm.processed_images[color_index],
                           cmap=plt.cm.gray)
            axes[i].set_axis_off()

    for i, ndcm in enumerate(subject.get_ndcms()):
        label = dcm_labels.get(ndcm, Label.UNLABELED).display_name()
        dcm = ndcm.get_data()
        if hasattr(dcm, 'SliceLocation'):
            axes[i].set_title('{} [{:.4f}]'.format(
                label, dcm.SliceLocation), fontsize='small', snap=True)
        else:
            axes[i].set_title('{}'.format(label), fontsize='small', snap=True)

    fig.canvas.set_window_title(get_window_title())
    fig.canvas.draw()


def update():
    # save labels to label_data.npy
    global dcm_labels
    save_dcm_labels(dcm_labels, verbal=False)

    update_plot()


def on_key_press(event):
    global dcm_labels, subjects, subject_index, displayed_subject, current_label
    global color_index, color_labels

    if event.key == 'left':
        subject_index -= 1
    if event.key == 'right':
        subject_index += 1

    if event.key == 'h':
        subject_index -= 10
    if event.key == 'l':
        subject_index += 10

    if event.key == 't':
        color_index += 1
        color_index %= len(color_labels)

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
            ndcm = subject.get_ndcms()[i]
            dcm_labels[ndcm] = current_label

    update()


def main():
    global dcm_dataset
    global subject_index, axes, subjects, dcm_labels, current_label
    global dicom_dict, dcm_labels, subjects, LABEL_FILE

    dcm_dataset = DicomDataset(data_directory=args.dir, preload_data=True)
    subjects = list(dcm_dataset.subjects.values())

    dcm_labels = load_dcm_labels(dcm_dataset)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_button_press)

    plt.subplots_adjust(top=0.95, bottom=0.05, right=1, left=0,
                        hspace=0.2, wspace=0)
    update()
    plt.show()

    save_dcm_labels(dcm_labels)

    # save as jpegs
    print('saving labeled images...')

    try:
        shutil.rmtree('cap_labeled')
    except FileNotFoundError:
        pass

    os._exit(1)


if __name__ == "__main__":
    main()