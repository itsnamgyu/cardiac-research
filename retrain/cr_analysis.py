import cr_interface as cri
import numpy as np
import pandas as pd
import sklearn as skl


metadata = cri.load_metadata()


class ResultDataFrame(pd.DataFrame):
    '''
    Columns: cr_code	dataset	pid	phase	slice	label	prediction
    '''
    @property
    def _constructor(self):
        return ResultDataFrame 

    def get_confusion_matrix(self):
        return pd.crosstab(self['label'], self['prediction'])

    def get_precision_and_recall(self):
        pd.crosstab(self['label'], self['prediction'])

        precision = {} # 정밀도
        recall = {} # 재현율

        lp = self[['label', 'prediction']]
        for label in set(self['label']):
            true_positives = lp[lambda e: (e.label == label) & (e.prediction == label)]
            truths = lp[lambda e: e.label == label]
            positives = lp[lambda e: e.prediction == label]

            precision[label] = len(true_positives) / len(positives)
            recall[label] = len(true_positives) / len(truths)

        return pd.DataFrame(dict(precision=precision,recall=recall))
    

def get_result_df(result, training=False):
    if training:
        cr_codes = map(cri.extract_cr_code, result['training_images'])
    else:
        cr_codes = map(cri.extract_cr_code, result['predictions'].keys())
        
    cr_codes = list(set(cr_codes))
    
    for i, cr_code in enumerate(cr_codes):
        cr_codes[i] = 'D00' + cr_code[3:]

    extract_dataset = lambda cr_code: cri.parse_cr_code(cr_code)[0]
    extract_pid = lambda cr_code: cri.parse_cr_code(cr_code)[1]
    extract_phase = lambda cr_code: cri.parse_cr_code(cr_code)[2]
    extract_slice = lambda cr_code: cri.parse_cr_code(cr_code)[3]

    datasets = list(map(extract_dataset, cr_codes))
    pids = list(map(extract_pid, cr_codes))
    phases = list(map(extract_phase, cr_codes))
    slices = list(map(extract_slice, cr_codes))

    extract_label = lambda cr_code: metadata[cr_code]['label']
    labels = list(map(extract_label, cr_codes))
    
    # make tri labels
    for i, label in enumerate(labels):
        if label in ['ap', 'bs', 'md']:
            labels[i] = 'in'
        
    d = dict(
            cr_code=cr_codes,
            dataset=datasets,
            pid=pids,
            phase=phases,
            slice=slices,
            label=labels,
    )
    
    if not training:
        extract_prediction = lambda cr_code: result['predictions']['testing_{}'.format(cr_code)]['prediction']
        predictions = list(map(extract_prediction, cr_codes))
        d['prediction'] = predictions
        

    return ResultDataFrame(d)
