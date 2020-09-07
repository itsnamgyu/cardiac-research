# Loader interface for CAP challenge images

import sys

import os
import glob
import re
from typing import List, Dict

import pydicom
from natsort import natsorted

import cr_interface as cri
import cr_importer


LOAD_PHASE_LIST = [0, 14]
IMPORT_DIR = 'CAP_challenge_training_set'
DATASET_INDEX = 0
# IMPORT_DIR = 'CAP Validation'
# DATASET_INDEX = 1


class DataImporter(cr_importer.DataImporter):
    def __init__(self, import_path=IMPORT_DIR, phase=0):
        self.import_path = os.path.join(cri.DATASET_DIR, import_path)
        self.phase = phase
        super().__init__()

    DEFAULT_DCM_PATH_FORMAT = '**/DET*SA*ph*.dcm'

    @staticmethod
    def get_filtered_dcm_paths(directory, path_format=DEFAULT_DCM_PATH_FORMAT) -> List[str]:
        if not os.path.isdir(directory):
            print('Dicom dataset directory {} does not exist'.format(directory))
            return []

        dcms: List[str] = []
        dcms += glob.glob(os.path.join(directory, path_format), recursive=True)
        dcms = natsorted(dcms)
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

            if phase_index == self.phase:
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
    for phase in LOAD_PHASE_LIST:  # hotfix for batch multi-phase import
        importer = DataImporter(phase=phase)
        importer.import_data(dataset_index=DATASET_INDEX, extension=".jpg")


if __name__ == "__main__":
    main()
