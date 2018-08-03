# Temporary module to summarize results


import os
import argparse
import collections
import os.path as op
import glob
import json
import copy

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import cr_interface as cri
import scipy.misc
import scipy.ndimage


LABELS = [None, 'oap', 'ap', 'md', 'bs', 'obs']

DISPLAY_NAME = {
    'oap': 'OUT_AP',
    'ap': 'AP',
    'md': 'MID',
    'bs': 'BS',
    'obs': 'OUT_BS',
    'nan': '-'
}


def generate_dataset_collections():
    '''
    Returns
    {
        'training': { 
            'label': {
                'patient_id': {
                    [ 'cr_code', ... ]
                }
            }

        }
    '''
    training_images = defaultdict(dict)
    testing_images = defaultdict(dict)

    IMAGES_DIR = 'images'
    image_paths = glob.glob(op.join(IMAGES_DIR, '**/*.aug.jpg'))
    image_paths.extend(glob.glob(op.join(IMAGES_DIR, '**/testing*.jpg')))
    # 'images/obs/training_D00_P00026901_P00_S10_CP20_R005.aug.jpg

    for path in image_paths:
        label = op.basename(op.dirname(path))
        cr_code = cri.extract_cr_code(path)
        parsed = cri.parse_cr_code(cr_code)
        patient_index = parsed[1]

        if 'training' in path:
            collection = training_images
        else:
            collection = testing_images

        # Nested defaultdict too tricky - default to None checking
        if patient_index not in collection[label]:
            collection[label][patient_index] = list()
        if cr_code not in collection[label][patient_index]:
            collection[label][patient_index].append(cr_code)

    return training_images, testing_images


def summarize_collection(collection):
    '''
    Returns
    {
        'all': {
            'patient_count': int,
            'image_count': int,
        }
        'label': {
            'image_count': int,
            'image_per_patient': int,
            'ratio: float,
        }
        ...
    }
    '''
    summary = defaultdict(dict)

    patient_count = 0
    total_image_count = 0
    for label, patient_dict in collection.items():
        patient_count = max(len(patient_dict), patient_count)
        image_count = 0
        for patient_index, images in patient_dict.items():
            image_count += len(images)

        total_image_count += image_count
        summary[label]['image_count'] = image_count

    summary['all']['patient_count'] = patient_count
    summary['all']['image_count'] = total_image_count

    for label, label_dict in summary.items():
        if label != 'all':
            label_dict['ratio'] = label_dict['image_count'] / total_image_count
            label_dict['image_per_patient'] = label_dict['image_count'] / patient_count

    return summary


def is_tri_label_result(result_dict):
    for image_dict in result_dict['predictions'].values():
        if image_dict['truth'] == 'ap':
            return False

    return True


def get_module_name(module_url, allow_exception=True):
    module_dict = {
        'nasnet': 'https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1',
        'mobilenet': 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1',
        'inception': 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1',
        'inception_resnet': 'https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1',
        'resnet': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
    }

    for name, url in module_dict.items():
        if module_url == url:
            return name

    return module_url


def load_modded_results(tri_label=True):
    '''
    Returns
    [
	{
            'training_steps': int
            'learning_rate': float
            'validation_percentage': float
            'batch_size': int
            'test_accuracy': float
            'bilabel_test_accuracy': float *
            'module_name': module name (instead of url!) *
        }, ...
    ]

    *: modded fields

    '''
    results = cri.load_results()
    modded_results = []

    if tri_label:
        results = [r for r in results if is_tri_label_result(r)]
    else:
        results = [r for r in results if not is_tri_label_result(r)]

    for result in results:
        modded_result = copy.deepcopy(result)

        # Calulate bilabel accuracy (did it accurately predict whether it's in/out)
        correct = 0
        total = 0
        for image_result in modded_result['predictions'].values():
            p = image_result['prediction']
            t = image_result['truth']

            if tri_label:
                inner_labels = ['in']
            else:
                inner_labels = ['ap', 'md', 'bs']

            if (p in inner_labels) == (t in inner_labels):
                correct += 1
            total += 1
        modded_result['bilabel_test_accuracy'] = correct / total
        modded_result['module_name'] = get_module_name(modded_result['tfhub_module'])

        del modded_result['training_images']
        del modded_result['predictions']
        del modded_result['tfhub_module']

        modded_results.append(modded_result)

    return modded_results


def main():
    metadata = cri.load_metadata()

    # Get images by label
    training_set, testing_set = generate_dataset_collections()


    # Get summaries
    training_set_summary = summarize_collection(training_set)
    testing_set_summary = summarize_collection(testing_set)

    print('{:-^80}'.format('Training Set'))
    print(json.dumps(training_set_summary, indent=4))

    print('{:-^80}'.format('Test Set'))
    print(json.dumps(testing_set_summary, indent=4))

    # Load Results w/ Bilabel Predictions (Tri-label learning)
    results = load_modded_results(tri_label=True)
    results.sort(key=lambda r: r['test_accuracy'], reverse=True)

    print('{:-^80}'.format('Learning Results (Tri-Label)'))
    for i, result in enumerate(results):
        print()
        print('%-5dModule: %-20sSteps: %-10sRate: %-10sAccuracy: %-10.4fBi-Accuracy: %-10.4f' % (
            i,
            result['module_name'],
            result['training_steps'],
            result['learning_rate'],
            float(result['test_accuracy']),
            result['bilabel_test_accuracy'],
        ))

    # Load Results w/ Bilabel Predictions (5-label learning)
    results = load_modded_results(tri_label=False)
    results.sort(key=lambda r: r['test_accuracy'], reverse=True)

    print()
    print('{:-^80}'.format('Learning Results (5-Label)'))
    for i, result in enumerate(results):
        print()
        print('%-5dModule: %-20sSteps: %-10sRate: %-10sAccuracy: %-10.4fBi-Accuracy: %-10.4f' % (
            i,
            result['module_name'],
            result['training_steps'],
            result['learning_rate'],
            float(result['test_accuracy']),
            result['bilabel_test_accuracy'],
        ))

if __name__ == "__main__":
    main()
