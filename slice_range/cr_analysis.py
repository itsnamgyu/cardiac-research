import json
import os
import warnings
import errno

import numpy as np
import pandas as pd
import keras

import cr_interface as cri


metadata = cri.load_metadata()


class Result():
    def __init__(self, data: dict):
        self.data = data
        self.df = Result._generate_dataframe(data)

    @classmethod
    def from_json(cls, dirname, basename='cr_result.json', full_path=False):
        if full_path:
            path = os.path.join(dirname, basename)
        else:
            path = os.path.join(cri.RESULTS_DIR, dirname)
            path = os.path.join(path, basename)

        with open(path) as f:
            data = json.load(f)

        return cls(data)

    @classmethod
    def from_predictions(
            cls, predictions, cr_codes, params, short_name,
            description='', classes=['in', 'oap', 'obs']):
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

        CORE_PARAMS = ['epochs', 'lr']

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
            d['truth'] = metadata[cr_codes[i]]['label']
            d['prediction'] = classes[p]
            d['percentages'] = dict(zip(
                classes, list(map(lambda f: str(f), prediction_vector))))

            results['predictions'][cr_code]=d

            if d['prediction'] == d['truth']:
                answers += 1

        results['test_accuracy']=str(answers / len(predictions))
        results['params']=params
        results['short_name']=short_name
        results['description']=description

        return cls(results)

    @classmethod
    def _generate_dataframe(cls, result: dict, tri_label=True):
        cr_codes=result['predictions'].keys()

        extractors=dict(
                dataset=lambda cr_code: cri.parse_cr_code(cr_code)[0],
                pid=lambda cr_code: cri.parse_cr_code(cr_code)[1],
                phase=lambda cr_code: cri.parse_cr_code(cr_code)[2],
                slice=lambda cr_code: cri.parse_cr_code(cr_code)[3],
                prediction=lambda cr_code: result['predictions'][cr_code]['prediction'],
                truth=lambda cr_code: result['predictions'][cr_code]['truth'],
        )

        result_dict=dict()
        for key, extractor in extractors.items():
            result_dict[key]=list(map(extractor, cr_codes))

        if tri_label:
            labels = result_dict['truth']
            for i, label in list(enumerate(labels)):
                if label in ['ap', 'bs', 'md']:
                    labels[i] = 'in'

        return pd.DataFrame(result_dict)

    def to_json(self, dirname, basename='cr_result.json', full_path=None) -> None:
        if full_path:
            path = os.path.join(dirname, basename)
        else:
            path = os.path.join(cri.RESULTS_DIR, dirname)
            path = os.path.join(path, basename)

        try:
            os.makedirs(os.path.dirname(path))
        except OSError as e:
            if e.errno == errno.EEXIST:
                warnings.warn('results directory already exists')
                print(e)
            else:
                raise

        with open(path, 'w') as f:
            json.dump(self.data, f)

    def get_accuracy(self) -> float:
        tp = self.df[['truth', 'prediction']]
        correct = tp[lambda e: (e.truth == e.prediction)]
        accuracy = len(correct) / len(tp)

        return accuracy

    def get_soft_accuracy(self) -> float:
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
        raise NotImplementedError()

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

            precision[truth] = len(true_positives) / len(positives)
            recall[truth] = len(true_positives) / len(truths)

        return pd.DataFrame(dict(precision=precision, recall=recall))

    def describe(self) -> None:
        '''
        Convinience function that prints core metrics
        '''
        string = ''
        string += '{:<18s}: {}\n'.format('Model', self.data['short_name'])
        string += '{:<18s}: {}\n'.format('Accuracy', self.get_accuracy())
        #string += '{:<18s}: {}\n\n'.format('Soft Accuracy', self.get_soft_accuracy())
        string += '{:<18s}: {}\n\n'.format('Soft Accuracy', 0)
        string += str(self.get_confusion_matrix()) + '\n\n'
        string += str(self.get_precision_and_recall()) + '\n\n'

        return string



def evaluate_model(model: keras.models.Model):
    '''
    Convinience function to quickly evaluate model performance
    '''
