import errno
import glob
import json
import os
import traceback
import warnings
import argparse

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import cr_interface as cri
from core import BASE_DIR
from core import paths

metadata = cri.load_metadata()

DEFAULT_CLASSES = ['in', 'oap', 'obs']
RESULTS_SCHEMA_VERSION = '2.0'

# params that should be passed manually while calling from_predictions
CORE_PARAMS = ['epochs', 'lr']

results = None
currently_loaded_results_dir = None


class Result():
    def __init__(self, data: dict):
        self.data = data
        self.df = Result._generate_dataframe(data)

    @staticmethod
    def _key_to_dir(key):
        warnings.warn('Deprecated')
        if isinstance(key, str):
            dirname = key
        else:
            try:
                iterator = iter(key)
            except TypeError:
                raise TypeError('keys must be a string or iterable of strings')
            else:
                dirname = os.path.join(*iterator)

        return dirname

    @classmethod
    def load(cls, model_key, instance_key, exp_key=None):
        if not exp_key:
            exp_key = paths.get_exp_key_from_dir('.')
        path = paths.get_test_result_path(exp_key, model_key, instance_key)
        return cls.load_from_path(path)

    def save(self, model_key, instance_key, exp_key=None):
        if not exp_key:
            exp_key = paths.get_exp_key_from_dir('.')
        path = paths.get_test_result_path(exp_key, model_key, instance_key)
        return self.save_to_path(path)

    @classmethod
    def load_from_path(cls, path):
        with open(path) as f:
            data = json.load(f)
        result = cls(data)

        # update legacy result files
        if 'schema_version' not in result.data:
            result.data['schema_version'] = '1.0'
        if result.data['schema_version'] != RESULTS_SCHEMA_VERSION:
            result.data['schema_version'] = RESULTS_SCHEMA_VERSION
            result.populate_metrics()
            result.to_json(key, basename, full_path, overwrite=True)

        return result

    def save_to_path(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.data, f)

    @staticmethod
    def select():
        return select_result()

    @classmethod
    def from_predictions(cls,
                         predictions,
                         cr_codes,
                         params,
                         short_name,
                         description='',
                         classes=DEFAULT_CLASSES,
                         tri_label=True):
        '''
        Generate a Result class from cr_codes and their respective predictions
        Note that you must use the `save_as_json` method to save the results as
        `cr_results.json`.

        # Arguments
        - predictions: predictions made by model in one-hot form. You can use
          the output from keras.Model.predict. Should be ndarray-ish with shape of
          (N, len(classes)).
        - cr_codes: list of N cr_codes in the same order as predictions.
        - classes: the classes corresponding to the indexes of the one-hot-ish vectors
          in predictions
        - params: params that describe the model from which the predictions came from
        - short_name: the identifier for the result
        - description: a verbose description of this results including dataset, 
          method, date etc.

        # Returns
        Result instance
        '''
        global metadata

        if not set(CORE_PARAMS) <= set(params.keys()):
            warnings.warn("params doesn't contain {}".format(CORE_PARAMS))

        if len(predictions) != len(cr_codes):
            raise ValueError(
                'lengths of predictions and cr_codes do not match')

        results = dict()
        results['predictions'] = {}
        answers = 0

        for i, p in enumerate(np.argmax(predictions, axis=1)):
            cr_code = cr_codes[i]
            prediction_vector = predictions[i]
            d = dict()
            if tri_label:
                d['truth'] = cri.to_tri_label(metadata[cr_codes[i]]['label'])
            else:
                d['truth'] = metadata[cr_codes[i]]['label']
            d['prediction'] = classes[p]
            d['percentages'] = dict(
                zip(classes, list(map(lambda f: str(f), prediction_vector))))

            results['predictions'][cr_code] = d

            if d['prediction'] == d['truth']:
                answers += 1

        results['test_accuracy'] = str(answers / len(predictions))
        results['params'] = params
        results['short_name'] = short_name
        results['description'] = description
        results['schema_version'] = RESULTS_SCHEMA_VERSION

        result = cls(results)
        result.populate_metrics()

        return result

    def populate_metrics(self):
        metrics = dict()
        if 'metrics' not in self.data:
            self.data['metrics'] = metrics

        metrics['accuracy'] = self._get_accuracy()
        metrics['soft_accuracy'] = self._get_soft_accuracy()
        metrics.update(self.get_auc_f1())

    def get_accuracy(self):
        return self.data['metrics']['accuracy']

    def get_soft_accuracy(self):
        return self.data['metrics']['soft_accuracy']

    def get_confusion_matrix(self):
        return pd.crosstab(self.df['truth'], self.df['prediction'])

    def get_precision_and_recall(self):
        pd.crosstab(self.df['truth'], self.df['prediction'])

        precision = {}
        recall = {}

        tp = self.df[['truth', 'prediction']]
        for truth in set(self.df['truth']):
            true_positives = tp[lambda e:
                                (e.truth == truth) & (e.prediction == truth)]
            truths = tp[lambda e: e.truth == truth]
            positives = tp[lambda e: e.prediction == truth]

            try:
                precision[truth] = len(true_positives) / len(positives)
            except ZeroDivisionError:
                precision[truth] = float('nan')
            try:
                recall[truth] = len(true_positives) / len(truths)
            except ZeroDivisionError:
                recall[truth] = float('nan')

        return pd.DataFrame(dict(precision=precision, recall=recall))

    def get_auc_f1_by_class(self):
        keys = DEFAULT_CLASSES

        p = self.data['predictions'].values()
        truths = np.array(list(map(self._extract_truth_array, p)))
        percentages = np.array(list(map(self._extract_percentage_array, p)))
        predictions = np.array(list(map(self._extract_prediction_array, p)))

        auc = roc_auc_score(truths, percentages, average=None)
        auc = dict(zip(keys, auc))
        f1 = f1_score(truths, predictions, average=None)
        f1 = dict(zip(keys, f1))

        return pd.DataFrame(dict(auc=auc, f1=f1))

    def get_auc_f1(self):
        p = self.data['predictions'].values()
        truths = np.array(list(map(self._extract_truth_array, p)))
        percentages = np.array(list(map(self._extract_percentage_array, p)))
        predictions = np.array(list(map(self._extract_prediction_array, p)))

        auc_micro = roc_auc_score(truths, percentages, average='micro')
        auc_macro = roc_auc_score(truths, percentages, average='macro')
        f1_micro = f1_score(truths, predictions, average='micro')
        f1_macro = f1_score(truths, predictions, average='macro')

        return dict(auc_micro=auc_micro,
                    auc_macro=auc_macro,
                    f1_micro=f1_micro,
                    f1_macro=f1_macro)

    def describe(self) -> str:
        '''
        Convinience function that prints core metrics
        '''
        string = ''
        string += '{:<18s}: {}\n'.format('Model', self.data['short_name'])
        for key, val in self.data['metrics'].items():
            string += '{:<18s}: {:.6f}\n'.format(key, float(val))
        string += '\n'
        string += str(self.get_confusion_matrix()) + '\n\n'
        string += str(self.get_precision_and_recall()) + '\n\n'
        string += str(self.get_auc_f1_by_class()) + '\n'

        return string

    @classmethod
    def _generate_dataframe(cls, result: dict, tri_label=True):
        cr_codes = result['predictions'].keys()

        extractors = dict(
            dataset=lambda cr_code: cri.parse_cr_code(cr_code)[0],
            pid=lambda cr_code: cri.parse_cr_code(cr_code)[1],
            phase=lambda cr_code: cri.parse_cr_code(cr_code)[2],
            slice=lambda cr_code: cri.parse_cr_code(cr_code)[3],
            prediction=lambda cr_code: result['predictions'][cr_code][
                'prediction'],
            truth=lambda cr_code: result['predictions'][cr_code]['truth'],
        )

        result_dict = dict()
        for key, extractor in extractors.items():
            result_dict[key] = list(map(extractor, cr_codes))

        if tri_label:
            labels = result_dict['truth']
            for i, label in list(enumerate(labels)):
                if label in ['ap', 'bs', 'md']:
                    labels[i] = 'in'

        return pd.DataFrame(result_dict)

    @staticmethod
    def _extract_truth_array(prediction, keys=DEFAULT_CLASSES):
        array = []
        for cls in keys:
            if (cls == prediction['truth']):
                array.append(1.0)
            else:
                array.append(0.0)
        return array

    @staticmethod
    def _extract_percentage_array(prediction, keys=DEFAULT_CLASSES):
        array = []
        for cls in keys:
            array.append(float(prediction['percentages'][cls]))
        return array

    @staticmethod
    def _extract_prediction_array(prediction, keys=DEFAULT_CLASSES):
        array = []
        for cls in keys:
            if (cls == prediction['prediction']):
                array.append(1.0)
            else:
                array.append(0.0)
        return array

    def _get_accuracy(self) -> float:
        tp = self.df[['truth', 'prediction']]
        correct = tp[lambda e: (e.truth == e.prediction)]
        try:
            accuracy = len(correct) / len(tp)
        except ZeroDivisionError:
            warnings.warn('there are no predictions in result {}'.format(
                self.data['short_name']))
            accuracy = float('nan')

        return accuracy

    def _get_soft_accuracy(self) -> float:
        '''
        The standards for labeling cardiac short-axis MRI images leave
        room for some ambiguity. One practitioner may determine a given slice
        as apical while another may see it as middle. To cope for this
        interobserver variability, we use another metric called soft accuracy.
        In this metric, we consider slices that border two different sections to be
        in a gray area. Thus predicting it to be on either side of the border is
        considered to be a correct assessment. Here is an example that illustrates the
        gray-area-slices.

        Prescribed labels: O O O A A A A M M M M B B B B B B O O O
        With gray areas:   O O G G A A G G M M G G B B B B G G O O
        '''
        df = self.df.copy().sort_values(['pid', 'phase',
                                         'slice']).reset_index(drop=True)

        answers = 0
        for i, row in df.iterrows():
            if df.loc[i, 'prediction'] == df.loc[i, 'truth']:
                answers += 1
                continue

            if i != 0:
                j = i - 1
                row_i = df.loc[i]
                row_j = df.loc[j]
                if row_i['pid'] == row_j['pid'] and\
                    row_i['phase'] == row_j['phase'] and\
                    row_i['prediction'] == row_j['truth']:
                    answers += 1
                    continue

            if i + 1 != len(df):
                j = i + 1
                row_i = df.loc[i]
                row_j = df.loc[j]
                if row_i['pid'] == row_j['pid'] and\
                    row_i['phase'] == row_j['phase'] and\
                    row_i['prediction'] == row_j['truth']:
                    answers += 1
                    continue

        try:
            soft_accuracy = answers / len(df)
        except ZeroDivisionError:
            warnings.warn('there are no predictions in result {}'.format(
                self.data['short_name']))
            soft_accuracy = float('nan')

        return soft_accuracy


def get_results(results_dir=cri.RESULTS_DIR, force_reload=False):
    global results
    global currently_loaded_results_dir
    if results is None or results_dir != currently_loaded_results_dir or force_reload:
        results = dict()
        result_paths = glob.glob(os.path.join(results_dir,
                                              '**/cr_result.json'),
                                 recursive=True)
        result_paths = map(lambda path: path[len(results_dir) + 1:],
                           result_paths)
        result_dirs = map(os.path.dirname, result_paths)

        for result_dir in result_dirs:
            try:
                result = Result.from_json(result_dir)
            except Exception as e:
                warnings.warn('invalid result file: {}'.format(
                    os.path.join(result_dir, 'cr_result.json')))
                print(traceback.format_exc())
                print(e)
                continue

            results[result_dir] = result

    return results


def export_csv(path='results.csv',
               full_path=False,
               results_dir=cri.RESULTS_DIR):
    if full_path:
        export_path = path
    else:
        export_path = os.path.join(cri.PROJECT_DIR, path)

    print('Exporting csv to {}'.format(export_path))

    metrics = [
        'accuracy',
        'soft_accuracy',
        'f1_macro',
        'auc_macro',
        'f1_micro',
        'auc_micro',
    ]

    results = get_results(results_dir, force_reload=True)
    result_pairs = results.items()
    result_pairs = sorted(result_pairs, key=lambda t: t[0])

    with open(export_path, 'w') as f:
        f.write('key,')
        f.write(','.join(metrics))
        f.write('\n')
        for key, result in result_pairs:
            f.write(key + ',')
            metric_values = [
                str(result.data['metrics'][metric]) for metric in metrics
            ]
            f.write(','.join(metric_values))
            f.write('\n')


def select_result(force_reload=False, return_key=False):
    results = get_results(results_dir=cri.RESULTS_DIR,
                          force_reload=force_reload)

    result_pairs = results.items()
    result_pairs = sorted(result_pairs, key=lambda t: t[0])

    print(' Results List '.center(80, '-'))
    print('  #. {:40s}{:10s}{:10s}'.format('KEY', 'ACC', 'F1_MACRO'))
    print('-' * 80)
    fmt = '{:3d}. {:40s}{:<10.4f}{:<10.4f}'
    for i, result_pair in enumerate(result_pairs):
        key, result = result_pair
        acc = result.data['metrics']['accuracy']
        f1_macro = result.data['metrics']['f1_macro']
        print(fmt.format(i, key, float(acc), float(f1_macro)))
    print('-' * 80)

    while True:
        try:
            index = int(input('Which of the results would you like to use? '))
            result_pair = result_pairs[index]
            if return_key:
                return result_pair
            else:
                return result_pair[1]
        except (IndexError, ValueError):
            print('Invalid index')
            continue


def evaluate_model(model: keras.models.Model,
                   input_data,
                   cr_codes,
                   classes=DEFAULT_CLASSES):
    '''
    Convenience function to quickly evaluate model performance
    '''
    predictions = model.predict(input_data)
    params = dict(epochs=0, lr=0)

    result = Result.from_predictions(predictions, cr_codes, params,
                                     'AUTO_EVAL', '')
    print(result.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    description = 'Select result and print details'
    parser.add_argument('-S',
                        '--select',
                        help=description,
                        action='store_true')
    args = parser.parse_args()

    if args.select:
        result = select_result()
        print(result.describe())
    else:
        export_csv()
