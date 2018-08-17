import cr_importer

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


# Loader interface for "Other Validation" images [TODO: change the name]


class DataImporter(cr_importer.DataImporter):
    def __init__(self, import_path='validate'):
        self.import_path = import_path
        super().__init__()

    def load_data_references(self) -> List[cr_importer.DataReference]:
        paths = glob.glob('validate/**/study/sax_*/IM-*-0001.dcm', recursive=True)
        paths = natsorted(paths)

        re_dcm = re.compile('validate\/([0-9]{3}).+')
        dcm_by_patient = defaultdict(list)
        for p in paths:
            match = re_dcm.search(p)
            patient_index = int(match.group(1))
            dcm_by_patient[patient_index].append(p)

        # data outliers
        dp = dcm_by_patient
        dp[516] = list(filter(lambda p: 'sax_104' not in p, dp[516]))
        dp[590] = list(filter(lambda p: '-0001-' in p, dp[590]))
        dp[597] = glob.glob('validate/597/**/*-0001-*.dcm', recursive=True)
        dp[619] = list(filter(lambda p: 'sax_21' not in p, dp[619]))
        def filter_56_57(p):
            if 'sax_56' in p or 'sax_57' in p:
                return '0001-0001' in p
            else:
                return True
        dp[623] = list(filter(filter_56_57, dp[623]))

        dcm_references: List[cr_importer.DcmDataReference] = []
        for pid, paths in dcm_by_patient.items():
            for i, path in enumerate(paths):
                dcm_references.append(cr_importer.DcmDataReference(
                    pid, i, 1, path, pydicom.dcmread(path)))

        return dcm_references


def main():
    importer = DataImporter()
    importer.import_data(dataset_index=2, test=False)


if __name__ == "__main__":
    main()
