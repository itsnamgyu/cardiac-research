import time
import requests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class Timer(object):
    def __init__(self, name=None):
        self.name= name

    def print_time(self, time):
        if self.name:
            print('Elapsed Time [{}]: {:,.3f}s'.format(self.name, time))
        else:
            print('Elapsed Time: {:,.3f}s'.format(time))

    def measure(self, f, repeat=1):
        start = time.time()
        for _ in range(repeat):
            f()
        elapsed = time.time() - start
        self.print_time(elapsed)

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exception_type, exception_value, traceback):
        elapsed = time.time() - self.start
        self.print_time(elapsed)


def notify(string='done'):
    # send a personal notification to Namgyu
    headers = {
        'Content-type': 'application/json',
    }
    data = '{"text":"[CREB1] %s"}' % string
    response = requests.post('https://hooks.slack.com/services/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', headers=headers, data=data)

def onehot(labels):
    # generate onehot labels from list of string labels
    labelize = LabelEncoder().fit_transform
    onehot = OneHotEncoder(sparse=False).fit_transform
    return onehot(labelize(list(labels)).reshape(-1, 1))
