import matplotlib.pyplot as plot
import numpy
import pandas
from .utils import train_test_split, fold_val_sets, mean_square_error


def main():
    data = pandas.read_csv('path/to/data/', sep='')
    period = numpy.atleast_2d(data['time'].values).T
    temperature = data['temp'].values

    Y = temperature
    X = period
    #
    #
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 0.4)

    degree = 15

    # F=Get reqularization constants
    lowest_err = float('inf')
    highes_reg_fact_ = None

    count = 10

    for reg_factor in numpy.arange(0, 0.1, 0.01):
        #
        #
        cross_val_sets = fold_val_sets(X_train, Y_train, count)

        mse = 0

        for _Xtrain, _Xtest, _Ytrain, _Ytest in cross_val_sets:
            data_model = PolynomialRRegression(
                degree=degree,
                reg_factor=reg_factor,
                learning_rate=0.01,
                iters=10000)
            data_model.fit(_Xtrain, _Ytrain)
            Y_prediction = data_model.predict(_Xtest)
            _mse = mean_square_error(_Ytest, _Y_prediction)
            mse += _mse
        mse /= count
