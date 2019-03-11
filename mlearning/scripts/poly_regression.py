"""
    Polynomial Regression
"""

import matplotlib.pyplot as plot
import numpy
import pandas

from ..supervised.regression import PolynomialRRegression
from ..helpers.utils.data_utils import data_helper

data_file = 'mlearning/data/time_temperature.txt'


def regress_polynomial():
    data = pandas.read_csv(data_file, sep='\t')
    period = numpy.atleast_2d(data['duration'].values).T
    temperature = data['temperature'].values

    Y = temperature
    X = period
    #
    #
    X_train, X_test, Y_train, Y_test = data_helper.split_train_test(X, Y, 0.4)

    degree = 15

    # Get reqularization constants
    lowest_err = float('inf')
    highes_reg_fact_ = None

    count = 10

    for reg_factor in numpy.arange(0, 0.1, 0.01):
        #
        #
        cross_val_sets = data_helper.fold_validation_set(
            X_train, Y_train, count)

        mse = 0
        print("Getting regularization const")
        for _Xtrain, _Xtest, _Ytrain, _Ytest in cross_val_sets:
            data_model = PolynomialRRegression(
                degree=degree,
                reg_factor=reg_factor,
                learning_factor=0.01,
                iters=10000)
            data_model.fit_constants(_Xtrain, _Ytrain)
            Y_prediction = data_model.make_prediction(_Xtest)
            _mse = data_helper.find_mse(_Ytest, Y_prediction)
            mse += _mse
        mse /= count
        print(f"Mean square root error: {mse} (regularization {reg_factor})")

        # Save constant with lowest error

        if mse < lowest_err:
            highest_reg_fact = reg_factor
            lowest_err = mse

        # Final prediction
        pred_model = PolynomialRRegression(degree=degree,
                                           reg_factor=highest_reg_fact,
                                           learning_factor=0.001,
                                           iters=10000)
        pred_model.fit_constants(X_train, Y_train)
        Y_prediction = pred_model.make_prediction(X_test)

        mse = data_helper.find_mse(Y_test, Y_prediction)
        print(f"Mean Squared Error: {lowest_err} from {highest_reg_fact}")

        Y_prediction_line = pred_model.make_prediction(X)

        # Plot data
        colour_map = plot.get_cmap('viridis')
        map1 = plot.scatter(366 * X_train, Y_train,
                            color=colour_map(0.9), s=10)
        map2 = plot.scatter(366 * X_test, Y_test, color=colour_map(0.5), s=10)
        plot.plot(366 * X, Y_prediction_line, color='black',
                  linewidth=2, label='Prediction')
        plot.suptitle("Polynomial Regression (Rg)")
        plot.title(f"MSE {mse}", fontsize=10)
        plot.get_cmap
        plot.xlabel('Day')
        plot.ylabel('Temperature in Celcius')
        plot.legend((map1, map2), ("Training Data",
                                   "Test Data"), loc='lower right')
        plot.show()


if __name__ == '__main__':
    regress_polynomial()
