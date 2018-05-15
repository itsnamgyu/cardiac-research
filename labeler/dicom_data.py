import glob
import re
import os
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pydicom
import argparse
import threading
from skimage import exposure
from typing import List, Dict, Tuple


class DicomData:
    re_dcm = re.compile("DET([0-9]+)_SA[0-9]+_ph[0-9]+.dcm")

    class Subject:
        def __init__(self, subject_id, dcm_names):
            self.dcm_names: List[str] = dcm_names
            self.dcms: List[pydcm.dataset.FileDataset] = []
            self.data_lock = threading.Lock()
            self.id = subject_id
            self.loaded = False

        def unload(self):
            with self.data_lock:
                del self.dcms[:]
                self.loaded = False

        def load(self):
            with self.data_lock:
                if not self.loaded:
                    for dcm_name in self.dcm_names:
                        self.dcms.append(pydicom.dcmread(dcm_name))
                    self.loaded = True

        def get_sorted_dcms(self):
            self.load()
            try:
                return sorted(self.dcms, key=lambda dcm : -dcm.SliceLocation)
            except:
                print('dicoms could not be sorted using SliceLocation')
                return self.dcms

    @staticmethod
    def get_filtered_dcm_names(base_directory) -> List[str]:
        def natural_key(string_):
            # From http://www.codinghorror.com/blog/archives/001018.html
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
        dcms: List[str] = []
        dcms += glob.glob(os.path.join(base_directory, "**/DET*SA*ph0.dcm"), recursive=True)
        dcms.sort(key=natural_key)
        return dcms

    def __init__(self, data_directory='cap_challenge', preload_data=False):
        self.subjects: Dict[int, Subject] = {}

        dcm_name_dict: Dict[int, List[str]] = {}
        for d in DicomData.get_filtered_dcm_names(data_directory):
            match = self.re_dcm.search(d)
            subject = int(match.group(1))
            if subject not in dcm_name_dict:
                dcm_name_dict[subject] = []
            dcm_name_dict[subject].append(d)

        for subject_id, dcm_names in dcm_name_dict.items():
            self.subjects[subject_id] = DicomData.Subject(subject_id, dcm_names)

        if preload_data:
            print('preloading dicom data... ', end='', flush=True)
            for subject in self.subjects.values():
                subject.load()
            print('done')

    def __str__(self):
        lines: List[str] = []
        lines.append('DicomData containing {} subjects'.format(len(self.subjects)))
        for subject in self.subjects.values():
            lines.append('Subject {:06d}: {:02d} Dicoms'.format(subject.id, len(subject.dcm_names)))
        return '\n'.join(lines)


def main():
    data = DicomData()
    print(data)

if __name__ == '__main__':
    main()
