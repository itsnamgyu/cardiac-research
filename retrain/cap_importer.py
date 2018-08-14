import cr_importer
from typing import List, Dict, Tuple

import os
import numpy as np
import glob
import scipy.misc
import pydicom
import re
from skimage import exposure


# Loader interface for CAP challenge images


class DataImporter(cr_importer.DataImporter):
    def __init__(self, import_path='cap_challenge_validation'):
        self.import_path = import_path
        super().__init__()

    DEFAULT_DCM_PATH_FORMAT = '**/DET*SA*ph0.dcm'

    @staticmethod
    def get_filtered_dcm_paths(directory, path_format=DEFAULT_DCM_PATH_FORMAT) -> List[str]:
        def natural_key(string_):
            # From http://www.codinghorror.com/blog/archives/001018.html
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

        if not os.path.isdir(directory):
            print('Dicom dataset directory {} does not exist'.format(directory))
            return []

        dcms: List[str] = []
        dcms += glob.glob(os.path.join(directory, path_format), recursive=True)
        dcms.sort(key=natural_key)
        return dcms

    def load_data_references(self) -> List[cr_importer.DataReference]:
        DCM_PATTERN = 'DET([0-9]+)_SA([0-9]+)_ph([0-9]+).dcm'
        re_dcm = re.compile(DCM_PATTERN)
        # { patient_index: [dcm, slice, phase, path], }
        self.patients: Dict[int, List] = {}

        patient_dict: Dict[int, pydicom.dataset.Dataset] = {}
        filtered_paths = DataImporter.get_filtered_dcm_paths(self.import_path)

        for dcm_path in filtered_paths:
            match = re_dcm.search(dcm_path)
            patient_index = match.group(1)
            slice_index = match.group(2)
            phase_index = match.group(3)

            patient_index = int(patient_index)
            slice_index = int(slice_index)
            phase_index = int(phase_index)

            if patient_index not in patient_dict:
                patient_dict[patient_index] = []
            patient_dict[patient_index].append([pydicom.dcmread(dcm_path), slice_index,
                                                phase_index, dcm_path])

        dcm_references: List[cr_importer.DcmDataReference] = []
        for patient_index, patient in patient_dict.items():
            try:
                patient = sorted(
                    patient, key=lambda image_data: image_data[0].SliceLocation)
                for index, image_data in enumerate(patient):
                    image_data[1] = index
            except AttributeError:
                pass

            for d in patient:
                dcm_references.append(cr_importer.DcmDataReference(
                    patient_index, d[1], d[2], d[3], d[0]))

        return dcm_references


def main():
    importer = DataImporter()
    importer.import_data(dataset_index=1)


if __name__ == "__main__":
    main()
