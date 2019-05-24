"""
    Particle Swam Optimization
"""
from ..helpers.utils.data_utils import data_helper
from ..supervised.particle_swam_opt import ParticleSwamOptimizedNN
from ..helpers.utils.display import plot_dimensioner

from sklearn import datasets
import numpy as np


def evolve_pso():
    """
        Trains neural network using particle swam optimization
    """
    data = datasets.load_iris()
    X = data_helper.normalize(data.data)
    y = data_helper.categorize(data.target.astype('int'))

    X_train, X_test, y_train, y_test = data_helper.split_train_test(
        X, y, .4, seed=1)
    model = ParticleSwamOptimizedNN(population=2000,
                                    cognitive_weight=.8,
                                    social_weight=.8,
                                    inertia_weight=.8,
                                    max_velocity=4.5)
    model = model.evolve(X_train, y_train, n_gens=50)
    loss, acc = model.test_on_batch(X_test, y_test)

    y_pred = np.argmax(model.make_prediction(X_test), axis=1)
    plot_dimensioner.plot_in_two_d(X_test,
                                   y_pred,
                                   title='Particle Swam Optimized NN',
                                   accuracy=acc,
                                   legend_labels=range(y.shape[1]))
