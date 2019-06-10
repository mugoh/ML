#!/usr/bin/env python3

from mlearning.scripts.poly_regression import regress_polynomial
from mlearning.scripts.convolutional_neural_network import convolute
from mlearning.unsupervised.gen_adv_net import Generative_Adversarial_Net
from mlearning.scripts.rbm import start_restricted_bolz_machine as rbm
from mlearning.scripts.genetic_algorithm import genetic_algr
from mlearning.scripts.deep_q_net import start_deepq_net
from mlearning.scripts.dbscan import run_dbscan
from mlearning.scripts.evolved_nn import start_evolved_nn
from mlearning.scripts.k_means import cluster
from mlearning.scripts.svm import cluster_svm
# from mlearning.scripts.naive_bayes import classify_nv_bayes
from mlearning.scripts.pam import cluster_pam
from mlearning.scripts.particle_swam_opt import evolve_pso
from mlearning.scripts.dcgan import dcgan
from mlearning.scripts.perceptron import perceptron
from mlearning.scripts.multilayer_perceptron import m_perceptron

if __name__ == '__main__':
    # regress_polynomial()
    # convolute()
    # Generative_Adversarial_Net()
    # rbm()
    # genetic_algr()
    # start_deepq_net()
    # run_dbscan()
    # start_evolved_nn()
    # cluster()
    # cluster_svm()
    # classify_nv_bayes()
    # cluster_pam()
    # evolve_pso()
    # dcgan()
    # perceptron()
    m_perceptron()
