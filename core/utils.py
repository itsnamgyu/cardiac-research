import os
import warnings
import datetime

EXP_DIR_TOKEN = '.cr_exp_token'


def get_current_time_key():
    """
    Get current time in the following format: YYYYMMDD_HHMMSS_mmmmmm.
    mmmmmm refers to microsecond. Here is an example: 20190101_164901_000000.
    """
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def validate_exp_dir(exp_dir):
    """
    Validate that the directory is a valid base directory for experiments.
    Valid experiment directories should be manually marked by including an
    empty file whose basename is the value of EXP_DIR_TOKEN.
    """
    abs_dir = os.path.abspath(exp_dir)
    token_dir = os.path.join(abs_dir, EXP_DIR_TOKEN)

    valid = os.path.isdir(abs_dir) and os.path.isfile(token_dir)
    if not valid:
        message = '{} is not a valid experiment directory. Maybe you ran the '\
            'python script from the wrong directory? If you are certain that this '\
            'is a valid experiment directory, create this file in the directory: '\
            '".cr_exp_dir"'.format(abs_dir)
        raise Exception(message)


def remove_safe(path):
    """
    Use this method when you want to safely remove a file. This effectively
    makes a backup of the file and notifies the user, if it exists.
    """
    abspath = os.path.abspath(path)
    if os.path.exists(abspath):
        backup_path = abspath + '.' + get_current_time_key() + '.bak'
        os.rename(abspath, backup_path)
        warnings.warn('Backing up "{}" as "{}"'.format(abspath, backup_path))
