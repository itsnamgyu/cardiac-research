import glob
import re
import os
import numpy as np
import pydicom
import threading
from skimage import exposure
from typing import List, Dict


class DicomDataset:
    DEFAULT_DCM_PATTERN = 'DET([0-9]+)_SA[0-9]+_ph[0-9]+.dcm'
    DEFAULT_DCM_PATH_FORMAT = '**/DET*SA*ph0.dcm'

    class NamedDicom:
        def equalize(pixel_array):
            return exposure.equalize_adapthist(pixel_array, clip_limit=0.03)

        def scale_color(pixel_array):
            p2, p98 = np.percentile(pixel_array, (2, 98))
            return exposure.rescale_intensity(pixel_array, in_range=(p2, p98))

        def __init__(self, dcm_path, re_dcm):
            self.dcm_path: str = dcm_path
            self.data_lock = threading.Lock()
            self.dcm_data = None
            self.name = os.path.splitext(os.path.basename(dcm_path))[0]
            self.processed_images = []

            match = re_dcm.search(self.dcm_path)
            self.subject_id = int(match.group(1))

        def get_data(self):
            if not self.dcm_data:
                self.load()
            return self.dcm_data

        def unload(self):
            with self.data_lock:
                self.dcm_data = None
                del self.processed_images[:]

        def load(self):
            with self.data_lock:
                if not self.dcm_data:
                    self.dcm_data = pydicom.dcmread(self.dcm_path)
                    img = self.dcm_data.pixel_array
                    self.processed_images.append(img)
                    self.processed_images.append(DicomDataset.NamedDicom.scale_color(img))
                    self.processed_images.append(DicomDataset.NamedDicom.equalize(img))

        def is_loaded(self):
            return self.dcm_data is not None

    class Subject:
        def __init__(self, subject_id, named_dcms):
            self.named_dcms = named_dcms
            self.sorted_named_dcms = None
            self.id = subject_id
            self.loaded = False
            self.sorted = False

        def unload(self):
            if self.loaded:
                for named_dcm in self.named_dcms:
                    named_dcm.unload()
                self.sorted = False
                self.loaded = False

        def load(self):
            if not self.loaded:
                for named_dcm in self.named_dcms:
                    named_dcm.load()
                self.loaded = True
                try:
                    self.named_dcms = sorted(self.named_dcms, key=lambda ndcm: -ndcm.get_data().SliceLocation)
                    self.sorted = True
                except AttributeError:
                    print('Patient #{} has no SliceLocation info; could not sort dcms.'.format(self.id))
                    self.sorted = False

        def get_ndcms(self):
            self.load()
            return self.named_dcms

    @staticmethod
    def get_filtered_dcm_paths(base_directory, path_format=DEFAULT_DCM_PATH_FORMAT) -> List[str]:
        def natural_key(string_):
            # From http://www.codinghorror.com/blog/archives/001018.html
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
        dcms: List[str] = []
        dcms += glob.glob(os.path.join(base_directory, path_format), recursive=True)
        dcms.sort(key=natural_key)
        return dcms

    def __init__(self, data_directory='cap_challenge', preload_data=False, dcm_pattern=DEFAULT_DCM_PATTERN):
        self.re_dcm = re.compile(dcm_pattern)
        self.subjects: Dict[int, DicomDataset.Subject] = {}

        subject_dict: Dict[int, List[DicomDataset.NamedDicom]] = {}
        for dcm_path in DicomDataset.get_filtered_dcm_paths(data_directory):
            ndcm = DicomDataset.NamedDicom(dcm_path, self.re_dcm)
            if ndcm.subject_id not in subject_dict:
                subject_dict[ndcm.subject_id] = []
            subject_dict[ndcm.subject_id].append(ndcm)

        for subject_id, ndcms in subject_dict.items():
            self.subjects[subject_id] = DicomDataset.Subject(subject_id, ndcms)

        if preload_data:
            def worker():
                for subject in self.subjects.values():
                    subject.load()
                print('preload complete')
            self.preload_thread = threading.Thread(target=worker)
            self.preload_thread.start()
            print('preloading dicom data in background...')
        else:
            self.preload_thread = None

    def __str__(self):
        lines: List[str] = []
        lines.append('DicomDataset containing {} subjects'.format(len(self.subjects)))
        for subject in self.subjects.values():
            lines.append('Subject {:06d}: {:02d} Dicoms'.format(subject.id, len(subject.named_dcms)))
        return '\n'.join(lines)


def main():
    data = DicomDataset(preload_data=False)
    print(data)


if __name__ == '__main__':
    main()
