import glob
import re
import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pydicom
import argparse


dicom_dict = {}
dicoms = []
label_dict = {}
subjects = []
subject_index = 0
loaded_subject = -1
axes = []

current_label = 0

dicoms = []
load_error = False
fig = plt.figure()


SUBJECT_DIRECTORY_FORMAT = "DET{:07d}"
LABEL_FILE = 'labels.npy'

LABEL_NAMES = [
        "UNLABELED",
        "OUT_OF_APICAL",
        "APICAL",
        "MIDDLE",
        "BASAL",
        "OUT_OF_BASAL"
        ]

LABEL_DISPLAY_NAMES = [
        "-",
        "OUT_AP",
        "AP",
        "MID",
        "BS",
        "OUT_BS"
        ]

"""
Group 0: Subject
Group 1: Index
Group 2: Phase (0 or 10)
"""
re_dicom = re.compile("DET([0-9]+)_SA([0-9]+)_ph([0-9]+).dcm")


def natural_key(string_):
    # From http://www.codinghorror.com/blog/archives/001018.html
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_filtered_dicoms(base_directory):
    dicoms = []
    dicoms += glob.glob(os.path.join(base_directory, "**/DET*SA*ph0.dcm"), recursive=True)
    dicoms.sort(key=natural_key)
    return dicoms


def get_filtered_dicom_dict(base_directory="cap_challenge"):
    """
    { subject0: [ index0_p0...],
      ...
    }
    """
    dicoms = get_filtered_dicoms(base_directory)
    dicom_dict = {}
    dicom_list = []
    subject_current = -1
    for d in dicoms:
        match = re_dicom.search(d)
        subject = int(match.group(1))
        index = int(match.group(2))
        phase = int(match.group(3))

        if subject_current == -1:
            subject_current = subject

        if subject != subject_current:
            dicom_dict[subject_current] = dicom_list
            dicom_list = []
            subject_current = subject

        dicom_list.append(d)

    dicom_dict[subject_current] = dicom_list

    return dicom_dict


def filter_dicoms_and_save(base_directory="cap_challenge", new_directory="cap_challenge_filtered"):
    dicom_dict = get_filtered_dicom_dict(base_directory)

    for key in dicom_dict:
        directory = SUBJECT_DIRECTORY_FORMAT.format(key)
        directory = os.path.join(new_directory, directory)

        try:
            os.makedirs(directory)
        except OSError:
            pass

        for d in dicom_dict[key]:
            shutil.copy2(d, directory)


def get_window_title():
    global load_error, current_label, subjects, subject_index

    subject = subjects[subject_index]
    if load_error:
        title = 'Error Loading Subject #{} [{}]'.format(subject_index, subject)
    else:
        title = 'Subject #{} [{}]'.format(subject_index, subject)
    title += ' [{}]'.format(LABEL_DISPLAY_NAMES[current_label])
    
    return title


def update_plot(replot):
    global subject_index, fig, axes, dicoms, dicom_dict, label_dict

    fig.canvas.set_window_title(get_window_title())

    if replot:
        fig.clf()
        del axes[:]
        for i, d in enumerate(dicoms):
            axes.append(fig.add_subplot(4, 6, i + 1))
            axes[i].imshow(dicoms[i].pixel_array, cmap=plt.cm.gray)
            axes[i].set_axis_off()

    for i, d in enumerate(dicoms):
        subject_name = subjects[subject_index]
        dicom_name = dicom_dict[subject_name][i]
        label = LABEL_DISPLAY_NAMES[int(label_dict[dicom_name][1])]
        axes[i].set_title(label, fontsize='small', snap=True)

    fig.canvas.draw()


def update_dicoms():
    global subject_index, loaded_subject, dicom_dict, load_error, dicoms

    if loaded_subject != subject_index:
        del dicoms[:]
        try:
            subject = subjects[subject_index]
            for dicom in dicom_dict[subject]:
                dicoms.append(pydicom.dcmread(dicom))
            load_error = False
        except OSError:
            print("could not load dicom file. was the data moved?")
            print("the labeling data is not affected so don't worry")
            load_error = True
        loaded_subject = subject_index
        return True

    return False


def update():
    replot = update_dicoms()
    update_plot(replot)


def on_key_press(event):
    global subject_index, subjects, current_label

    if event.key == 'left':
        subject_index -= 1
    if event.key == 'right':
        subject_index += 1

    if event.key == 'h':
        subject_index -= 10
    if event.key == 'l':
        subject_index += 10

    try:
        index = int(event.key)
        if 0 <= index <= len(LABEL_NAMES) - 1:
            current_label = index
    except ValueError:
        pass

    subject_index %= len(subjects)
    update()


def on_button_press(event):
    global index, fig, ax, TITLE
    global subject_index, axes, subjects, label_dict, dicom_dict, current_label

    fig.canvas.draw()

    subject_name = subjects[subject_index]
    for i, ax in enumerate(axes):
        if event.inaxes == ax:
            dicom_name = dicom_dict[subject_name][i]
            label_dict[dicom_name] = (LABEL_NAMES[current_label], str(current_label))

    update()


def ndarray_to_dict(ndarray):
    """
    ndarray:
    subject   label_name   label_index    var3    var4   ...
    ...

    dict: { subject: (label_name, label_index, var3, var4...), ... }
    """

    dictionary = {}
    for i in range(ndarray.shape[0]):
        dictionary[ndarray[i][0]] = tuple(ndarray[i][1:])
    return dictionary


def dict_to_ndarray(dictionary, ordered_keys=None):
    array = []
    if ordered_keys:
        for key in ordered_keys:
            array.append([key] + list(dictionary[key]))
    else:
        for subject, data in dictionary.items():
            array.append([subject] + list(data))

    return np.array(array)


def main():
    global dicom_dict, label_dict, subjects, LABEL_FILE

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dir', dest='dir',  type=str, nargs=1, default='cap_challenge',
            help='top directory of dicom files')

    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print('could not find data directory {}'.format(args.dir))
        print('terminating...')
        exit()

    dicom_dict = get_filtered_dicom_dict(args.dir)

    try:
        ndarray = np.load(LABEL_FILE)
        label_dict = ndarray_to_dict(ndarray)
        print('successfully retrieved labeling data from {}'.format('LABEL_FILE'))
    except OSError:
        print("could not find data file: {}".format(LABEL_FILE))
        print("initializing new data")
        label_dict = {}

    subjects = list(dicom_dict.keys())
    subjects.sort()

    for subject in subjects:
        for dicom in dicom_dict[subject]:
            if dicom not in label_dict:
                label_dict[dicom] = (LABEL_NAMES[0], 0)

    cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
    cid = fig.canvas.mpl_connect('button_press_event', on_button_press)

    update()
    plt.show()
    
    ndarray = dict_to_ndarray(label_dict)
    np.save(LABEL_FILE, ndarray)

    print('data successfully saved to {}'.format(LABEL_FILE))


if __name__ == "__main__":
    main()
