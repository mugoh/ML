import matplotlib.pyplot as plot
import numpy
import pandas
from .utils import train_test_split


def main():
    data = pandas.read_csv('path/to/data/', sep='')
    period = numpy.atleast_2d(data['time'].values).T
    temperature = data['temp'].values

    Y = temperature
    X = period

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 0.4)

    degree = 15
