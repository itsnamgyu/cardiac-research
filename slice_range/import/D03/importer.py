import cr_importer
import cr_interface as cri

import os
import re
from typing import List, Dict, Tuple
import glob
from collections import defaultdict

import pydicom
import numpy as np
import scipy.misc
from skimage import exposure
from natsort import natsorted


# Loader interface for CINE images from Samsung Medical Center [TODO: change the name]


class DataImporter(cr_importer.DataImporter):
    def __init__(self):
        super().__init__()

    def load_data_references(self) -> List[cr_importer.DataReference]:
        paths = glob.glob(os.path.join(cri.DATASET_DIR, 'cine/**/*img00001.dcm'),
                          recursive=True)
        paths = natsorted(paths)

        re_dcm = re.compile('cine/([0-9]{3}).+')
        dcm_by_patient = defaultdict(list)
        for p in paths:
            match = re_dcm.search(p)
            patient_index = int(match.group(1))
            dcm_by_patient[patient_index].append(p)

        dcm_references: List[cr_importer.DcmDataReference] = []
        for pid, paths in dcm_by_patient.items():
            for i, path in enumerate(paths):
                dcm_references.append(cr_importer.DcmDataReference(
                    pid, i, 1, path, pydicom.dcmread(path)))

        return dcm_references


def main():
    importer = DataImporter()
    importer.import_data(dataset_index=3)


if __name__ == "__main__":
    main()
