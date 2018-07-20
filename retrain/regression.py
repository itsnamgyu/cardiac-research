import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def regress(values):
    # values: [ 0, 2, 1, 3 ]
    # output: [ 0, 1, 2, 3 ]
    regr = linear_model.LinearRegression()
    regr.fit(np.arange(len(values)).reshape(-1 ,1), values)
    
    return regr.predict(np.arange(len(values)).reshape(-1, 1))


def main():
    regr = linear_model.LinearRegression()
    values = np.array([1, 4, 3, 8, 9, 13, 12, 15])
    regr.fit(np.arange(len(values)).reshape(-1 ,1), values)

    results = regr.predict(np.arange(len(values)).reshape(-1, 1))
    print('Results: \n', results)

    # Plot outputs
    plt.scatter(np.arange(len(values)), values,  color='black')
    plt.plot(np.arange(len(values)), results,  color='blue')

    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == '__main__':
    main()
