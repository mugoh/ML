import matplotlib.pyplot as plot
import numpy
import pandas
from .utils import train_test_split, fold_val_sets, mean_square_error


def regress_polynomial():
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
        print("Getting regularization const")
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
        print(f"Mean square root error: {mse} (regularization {reg_factor})")

        # Save constant with lowest error

        if mse < lowest_err:
            highest_reg_fact = reg_factor
            lowest_err = mse

        # Final prediction
        pred_model = PolynomialRRegression(degree,
                                           highest_reg_fact,
                                           0.001,
                                           10000)
        model.fit(X_train, Y_train)
        Y_prediction = pred_model.predict(X_test)

        mse = mean_square_error(Y_test, Y_prediction)
        print(f"Mean Squaredd Error: {lowest_err} from {highest_reg_fact}")

        Y_prediction_line = pred_model.predict(X)

        # Plot data
        colour_map = plot.get_cmap('viridis')
        map1 = plot.scatter(366 * X_train, Y_train,
                            color=colour_map(0.9), s=10)
        map2 = plot.scatter(366 * X_test, Y_test, color=colour_map(0.5), s=10)
        plot.plot(366 * X, Y_prediction_line, color='black',
                  linewidth=2, label='Prediction')
        plot.suptitle("Ploynomial Regression (Rg)")
        plot.title("MSE {mse}", fontsize=10)
        plot.xlabel = ('Day')
        plot.ylabel('Temperature in Celcius')
        plot.legend((map1, map2), ("Training Data",
                                   "Test Data"), loc='lower right')
        plot.show()


if __name__ == '__main__':
    regress_polnomial()
