import os

import pydicom
import scipy.misc
import matplotlib.pyplot as plt


"""dcm module
Utility functions to work with dicoms
"""


def plot_dicom(dcm: pydicom.dataset.Dataset) -> None:
    img = dcm.pixel_array
    return plt.imshow(img)


def save_dicom_as_image(dcm: pydicom.dataset.Dataset, path: str, mkdir=True) -> None:
    img = dcm.pixel_array
    if mkdir:
        os.system('mkdir -p {}'.format(os.path.dirname(path)))
    scipy.misc.imsave(path, img)
