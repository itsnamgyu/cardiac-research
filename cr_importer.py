import cr_interface as cri

import os
from typing import List
from abc import ABC, abstractmethod

import matplotlib as mpl
import matplotlib.image
import pydicom
from skimage import exposure
import numpy as np
import cv2



class DataReference(ABC):
    def __init__(self, patient_index: int, slice_index: int,
                 phase_index: int, original_filepath: str):
        self.patient_index: int = patient_index
        self.phase_index: int = phase_index
        # must be sequential (direction unspecified)
        self.slice_index: int = slice_index
        self.original_filepath: str = original_filepath
        name = os.path.basename(original_filepath)
        name = os.path.splitext(name)[0]
        self.original_name: str = name

    @abstractmethod
    def save_image(self, path: str) -> None:
        pass


class DcmDataReference(DataReference):
    def __init__(self, patient_index: int, slice_index: int, phase_index: int, original_filepath: str,
                 dcm: pydicom.dataset.Dataset):
        super().__init__(patient_index, slice_index, phase_index, original_filepath)
        self.dcm = dcm

    @staticmethod
    def scale_color(pixel_array):
        p2, p98 = np.percentile(pixel_array, (2, 98))
        return exposure.rescale_intensity(pixel_array, in_range=(p2, p98), out_range="uint16")

    def save_image(self, path: str) -> None:
        img = self.dcm.pixel_array
        img = DcmDataReference.scale_color(img)
        cv2.imwrite(path, img)


# Interface for loading modules
class DataImporter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_data_references(self) -> List[DataReference]:
        pass

    def import_data(self, dataset_index: int, test=False, extension='.jpg'):
        metadata = cri.load_metadata()

        # load using loader
        references = self.load_data_references()
        os.makedirs(cri.DATABASE_DIR, exist_ok=True)

        print('Processing {} images...'.format(len(references)))

        for i, r in enumerate(references):
            cr_code = cri.get_cr_code(dataset_index, r.patient_index,
                                      r.phase_index, r.slice_index)
            print('processing {}'.format(cr_code))

            if cr_code in metadata:
                print('{} already exists in database'.format(cr_code))
                continue

            path = os.path.join(cri.DATABASE_DIR, cr_code + extension)
            if test:
                print('\t'.join((cr_code, r.original_name, r.original_filepath, path)))
            else:
                metadata[cr_code] = {}
                metadata[cr_code]['original_filepath'] = r.original_filepath
                metadata[cr_code]['original_name'] = r.original_name
                r.save_image(path)

        cri.save_metadata(metadata)
