import sys
sys.path.append('../..')

import pandas as pd

import keras_utils as ku
import keras_history as kh


K = 5
EPOCHS = 5
LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001, 0.000001]


def load_average_history_by_lr(app, epochs=EPOCHS, k=K):
    histories = {}
    for lr_index, lr in enumerate(LEARNING_RATES):
        histories[lr] = kh.load_average_history(
            app.get_model(), lr_index, epochs, k=k)
    return histories


def plot_single_metric_average_history_by_lr(app, metric='val_acc', epochs=EPOCHS, k=K):
    h = load_average_history_by_lr(app=app, epochs=EPOCHS, k=5)
    single_metric = {}
    for lr, history in h.items():
        single_metric[lr] = history[metric]
    return pd.DataFrame(single_metric).plot(
        title='{}: {} by learning-rate'.format(app.name, metric),
        figsize=(10, 5),
    )

    
def generate_all_test_results():
    results = {}
    for app in ku.apps.values():
        try:
            results[app.codename] = kh.load_test_result(app.get_model())
            plot_single_metric_average_history_by_lr(app, 'val_acc').get_figure().savefig(
                '{}_val_acc_by_lr.png'.format(app.codename), dpi=160)
            plot_single_metric_average_history_by_lr(app, 'val_loss').get_figure().savefig(
                '{}_val_loss_by_lr.png'.format(app.codename), dpi=160)
        except Exception:
            pass
    results = pd.DataFrame(results)
    results.to_csv('test_results.csv')
    print(results)


generate_all_test_results()
