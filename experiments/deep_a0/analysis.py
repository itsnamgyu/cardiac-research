import matplotlib as mpl
import matplotlib.pyplot as plt
import traceback
import warnings
import os
import argparse

from core import history as ch
from core.fine_model import FineModel

_depth_key = 'D{:02}'
_fold_key = 'D{:02}_L{:02}_F{:02}'
_epoch_key = 'D{:02}_L{:02}_F{:02}_E{:03}'

from params import *

FIGURE_DIR = 'figures'

metric_names = {
    'val_loss': 'Validation Loss',
    'loss': 'Training Loss',
    'val_acc': 'Validation Accuracy',
    'acc': 'Training Accuracy',
}

bounds_per_metric = {
    'val_loss': (0, 2),
    'loss': (0, 2),
    'val_acc': (0.3, 1),
    'acc': (0.3, 1)
}


def _set_ax_bounds(ax, metric, force_bounds=None):
    bounds = force_bounds
    if bounds is None:
        if metric in bounds_per_metric:
            bounds = bounds_per_metric[metric]
        else:
            warnings.warn('bounds are not set for metric {}'.format(metric))
    ax.set_ylim(*bounds)


def plot_average_by_fold(histories,
                         title=None,
                         ax=None,
                         metric='val_loss',
                         force_bounds=None):
    """
    ax: matplotlib.Axes on which to plot the figure
    """
    if metric not in metric_names.keys():
        warnings.warn(
            'Metric "{}" not in metrics. Select one of the following: {}'.
            format(metric, metric_names.keys()))
        traceback.print_exc()
        fig = plt.Figure()
        return fig

    metric_name = metric_names[metric]

    _fold_label = 'Fold #{}'
    avg_label = 'K-Fold Average'
    figsize = (6, 6)
    fold_alpha = 0.5
    xlabel = 'Epochs'
    ylabel = metric_name

    if ax is None:
        fig, ax = plt.subplots(squeeze=True, figsize=figsize)

    if title is not None:
        ax.set_title(title)

    _set_ax_bounds(ax, metric, force_bounds)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i, history in enumerate(histories):
        ax.plot(history.index,
                history[metric],
                alpha=fold_alpha,
                label=_fold_label.format(i + 1))

    avg = ch.get_average(histories)[metric]
    ax.plot(avg, label=avg_label)
    ax.legend(loc='upper left')
    ax.grid()

    return ax


def plot_average_by_lr(histories_by_lr,
                       title=None,
                       ax=None,
                       metric='val_loss',
                       force_bounds=None):
    """
    histories_by_lr: {
        '0.001': list_of_history_dataframes,
        '0.0001': ...,
        ...,
    }
    ax: matplotlib.Axes on which to plot the figure
    """
    if metric not in metric_names.keys():
        warnings.warn(
            'Metric "{}" not in metrics. Select one of the following: {}'.
            format(metric, metric_names.keys()))
        traceback.print_exc()
        fig = plt.Figure()
        return fig

    metric_name = metric_names[metric]

    _fold_label = 'Fold #{}'
    figsize = (6, 6)
    xlabel = 'Epochs'
    ylabel = metric_name

    if ax is None:
        fig, ax = plt.subplots(squeeze=True, figsize=figsize)

    if title is not None:
        ax.set_title(title)

    _set_ax_bounds(ax, metric, force_bounds)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for lr, histories in histories_by_lr.items():
        avg_history = ch.get_average(histories)
        ax.plot(avg_history.index,
                avg_history[metric],
                label='{:.1E}'.format(lr))
    ax.legend(loc='upper left')
    ax.grid()

    return ax


def analyze_depth(fm,
                  verbose_model_name,
                  depth_index,
                  metric,
                  lr_list=LEARNING_RATES):
    """
    Plot training history for all learning rates + their average
    """
    model_key = fm.get_key()

    print('Analyzing {} D={}, LR={}'.format(verbose_model_name, depth_index,
                                            lr_list))

    title = verbose_model_name
    fig, axes = plt.subplots(1, len(lr_list), squeeze=True, figsize=(18, 6))
    axes = axes.flatten()
    histories_by_lr = dict()
    output_dir = os.path.join(
        FIGURE_DIR, '{model}_D{depth:02d}'.format(model=fm.get_key(),
                                                  depth=depth_index))
    os.makedirs(output_dir, exist_ok=True)
    for i, ax in enumerate(axes):
        lr = lr_list[i]
        histories = list()
        for k in range(K):
            fold_key = _fold_key.format(depth_index, i, k)
            history = ch.load_history(model_key, fold_key)
            if history is not None and not history.empty:
                histories.append(history)
        if len(histories):
            histories_by_lr[lr] = histories
            name = 'Fold {}.eps'.format(metric.upper())
            path = os.path.join(output_dir, name)
            plot_average_by_fold(histories,
                                 title='Learning Rate {:.1E}'.format(lr),
                                 metric=metric,
                                 ax=ax)
    fig.savefig(path,
                format='eps',
                dpi=320,
                bbox_inches='tight',
                transparent=False)

    lr_ax = plot_average_by_lr(histories_by_lr, title=title, metric=metric)
    name = 'Average {}.eps'.format(metric.upper())
    path = os.path.join(output_dir, name)
    lr_ax.get_figure().savefig(path,
                               format='eps',
                               dpi=320,
                               bbox_inches='tight',
                               transparent=False)


def analyze_lr(fm, verbose_model_name, depth_index, lr_index, lr_value,
               metric):
    """
    Plot training history for all K folds + their average
    """
    model_key = fm.get_key()

    print('Analyzing {} D={}, LR={}'.format(verbose_model_name, depth_index,
                                            lr_value))
    title = verbose_model_name
    histories_by_lr = dict()

    histories = list()
    histories_by_lr[lr_value] = histories
    for k in range(K):
        fold_key = _fold_key.format(depth_index, lr_index, k)
        history = ch.load_history(model_key, fold_key)
        if history is not None and not history.empty:
            histories.append(history)
    output_dir = os.path.join(
        FIGURE_DIR, '{model}_D{depth:02d}'.format(model=fm.get_key(),
                                                  depth=depth_index))
    os.makedirs(output_dir, exist_ok=True)
    name = 'Fold {} [LR={:.1E}].eps'.format(metric.upper(), lr_value)
    path = os.path.join(output_dir, name)
    ax = plot_average_by_fold(histories, title=title, metric=metric)
    ax.get_figure().savefig(path,
                            format='eps',
                            dpi=320,
                            bbox_inches='tight',
                            transparent=False)


def analyze_all(fm, verbose_model_name, depth_index, lr_list=None):
    if lr_list is None:
        warnings.warn('You should specify lr_list for analyze_all')
        lr_list = LEARNING_RATES

    metrics = metric_names.keys()
    for metric in metrics:
        print('Analyzing {} metric={}, depth_index={}'.format(
            verbose_model_name, metric, depth_index))
        analyze_depth(fm,
                      verbose_model_name,
                      lr_list=LEARNING_RATES,
                      depth_index=depth_index,
                      metric=metric)


if __name__ == '__main__':
    fm = FineModel.get_dict()['mobilenet_a25']()
    verbose_model_name = 'MobileNet(a=25)'
    analyze_all(fm, verbose_model_name, depth_index=0)
