from mlearning.scripts.poly_regression import regress_polynomial
from mlearning.scripts.convolutional_neural_network import convolute
from mlearning.unsupervised.gen_adv_net import Generative_Adversarial_Net
from mlearning.scripts.rbm import start_restricted_bolz_machine as rbm
from mlearning.scripts.genetic_algorithm import genetic_algr
from mlearning.scripts.deep_q_net import start_deepq_net

if __name__ == '__main__':
    # regress_polynomial()
    # convolute()
    # Generative_Adversarial_Net()
    # rbm()
    # genetic_algr()
    start_deepq_net()
