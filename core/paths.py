"""Manages experiment directories, output paths, etc.

Output directory format (example)

experiments/deep3/output/mobilenetv2/D00_L00_E000
|-- results.json
|-- train_val_history.json
|-- trained_weights.hd5

This output directory is comprised of three parts:
- Experiment key:   deep3
- Model key:        mobilenetv2
- Instance key:     D00_L00_E000

These keys form a single output key dict:
{
    experiment: 'deep3',
    model: 'mobilenetv2',
    instance: 'D00_L00_E000,
}
"""
import glob
import os
import warnings
import json

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_BASE_DIR = os.path.join(PROJECT_DIR, 'experiments')
OUTPUT_DIR = 'output'  # relative to EXP_DIR

WEIGHTS_BASENAME = 'trained_weights.hd5'
TEST_RESULT_BASENAME = 'test_result.json'
HISTORY_BASENAME = 'train_val_history.csv'


def get_exp_keys():
    return get_subdirectory_basenames(EXP_BASE_DIR)


def get_subdirectory_basenames(parent, only_directory=True):
    children = glob.glob(os.path.join(parent, '*'))
    if only_directory:
        children = filter(os.path.isdir, children)
    basenames = [os.path.relpath(c, parent) for c in children]
    return basenames


def get_model_keys(exp_key):
    exp_dir = os.path.join(EXP_BASE_DIR, exp_key, OUTPUT_DIR)
    return get_subdirectory_basenames(exp_dir)


def get_instance_keys(exp_key, model_key):
    model_dir = os.path.join(EXP_BASE_DIR, exp_key, OUTPUT_DIR, model_key)
    return get_subdirectory_basenames(model_dir)


def get_exp_key_from_dir(exp_dir=None):
    """Get experiment key from current directory. Used for scripts
    that are run from the experiment directories themselves.
    """
    if exp_dir is None:
        exp_dir = '.'
    abs_dir = os.path.abspath(exp_dir)
    dirname, basename = os.path.split(abs_dir)
    if dirname == EXP_BASE_DIR:
        return basename
    else:
        message = '{} is not a valid experiment directory. Experiment directories '\
            'should be under {}. Maybe you ran the python script from the wrong '\
            'directory?'.format(abs_dir, EXP_BASE_DIR)
        raise Exception(message)


def get_instance_path(exp_key, model_key, instance_key):
    """If exp_key is not specified, assume that the current directory
    is an experiment directory and infer the exp_key from the directory
    """
    if exp_key is None:
        exp_key = get_exp_key_from_dir()
    return os.path.join(EXP_BASE_DIR, exp_key, OUTPUT_DIR, model_key,
                        instance_key)


def get_weights_path(exp_key, model_key, instance_key):
    instance_path = get_instance_path(exp_key, model_key, instance_key)
    return os.path.join(instance_path, WEIGHTS_BASENAME)


def get_history_path(exp_key, model_key, instance_key):
    instance_path = get_instance_path(exp_key, model_key, instance_key)
    return os.path.join(instance_path, HISTORY_BASENAME)


def get_test_result_path(exp_key, model_key, instance_key):
    instance_path = get_instance_path(exp_key, model_key, instance_key)
    return os.path.join(instance_path, TEST_RESULT_BASENAME)


def test_result_exists(exp_key, model_key, instance_key):
    test_result_path = get_test_result_path(exp_key, model_key, instance_key)
    return os.path.isfile(test_result_path)


def get_output_tree():
    """{
        exp_key: {
            model_key: [
                instance_key_0,
                instance_key_1,
            ]
        }
    }
    """
    experiments = dict()
    for exp_key in get_exp_keys():
        models = dict()
        for model_key in get_model_keys(exp_key):
            instances = get_instance_keys(exp_key, model_key)
            if instances:
                models[model_key] = instances
        if len(models.values()):
            experiments[exp_key] = models
    return experiments


def select_output(condition=None) -> (str, str, str):
    """Interactive
    """
    print(' Select Output '.center(80, '-'))
    print('{:>3}   {:<20} {:<20} {:<20}'.format('IDX', 'EXP', 'MODEL',
                                                'INSTANCE'))
    fmt = '{:>3}   {:<20} {:<20} {:<20}'
    keys = []
    for e, models in get_output_tree().items():
        for m, instances in models.items():
            for i in instances:
                if not condition or condition(e, m, i):
                    print(fmt.format(len(keys), e, m, i))
                    keys.append((e, m, i))
    print('-' * 80)
    while True:
        choice = input('Enter selection (q to cancel): ')
        if choice.lower() == 'q':
            print('Quitting')
            return
        try:
            choice = int(choice)
            key = keys[choice]
            break
        except (ValueError, IndexError):
            print('Please enter a valid index')

    return key


def main():
    """Manual unit tests
    """

    print('Experiment Keys'.center(80, '-'))
    keys = get_exp_keys()
    print(keys)

    print()
    print('get_exp_from_dir Test'.center(80, '-'))
    test_key = keys[0]
    test_experiment_dir = os.path.join(EXP_BASE_DIR, test_key)
    print('Reverse key search match: {}'.format(
        test_key == get_exp_key_from_dir(test_experiment_dir)))

    print()
    print('Output Tree'.center(80, '-'))
    print(json.dumps(get_output_tree(), indent=4))


if __name__ == '__main__':
    main()
