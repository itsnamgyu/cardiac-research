import os
import re
import sys
from typing import List, Dict, Tuple
import glob
from collections import defaultdict

import pydicom
import numpy as np
import scipy.misc
from skimage import exposure
from natsort import natsorted

d = os.path.dirname
PROJECT_DIR = d(d(d(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)

import cr_importer
import cr_interface as cri



# Loader interface for CINE images from Samsung Medical Center [TODO: change the name]


class DataImporter(cr_importer.DataImporter):
    def __init__(self, import_path='validate'):
        self.import_path = import_path
        super().__init__()

    def load_data_references(self) -> List[cr_importer.DataReference]:
        paths = glob.glob(os.path.join(cri.DATASET_DIR, 'cine/**/*.dcm'),
                          recursive=True)
        paths = natsorted(paths)

        re_dcm = re.compile('cine/([0-9]{3}).+ser([0-9]{3})img([0-9]{5}).dcm')
        dcm_by_patient = defaultdict(list)
        for p in paths:
            match = re_dcm.search(p)
            patient_index = int(match.group(1))
            slice_index = int(match.group(2))
            phase_index = int(match.group(3))
            dcm_by_patient[patient_index].append(p)

        dcm_references: List[cr_importer.DcmDataReference] = []
        for pid, paths in dcm_by_patient.items():
            for path in paths:
                match = re_dcm.search(path)
                slice_index = int(match.group(2))
                phase_index = int(match.group(3))
                dcm_references.append(cr_importer.DcmDataReference(
                    patient_index=pid,
                    slice_index=slice_index,
                    phase_index=phase_index,
                    original_filepath=path,
                    dcm=pydicom.dcmread(path)))

        return dcm_references


def main():
    importer = DataImporter()
    importer.import_data(dataset_index=3)


if __name__ == "__main__":
    main()
