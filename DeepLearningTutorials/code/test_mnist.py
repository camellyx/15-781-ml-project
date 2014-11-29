import sys

import numpy
import scipy

import logistic_cg
import logisitc_sgd
import logistic_sgd_gaussian
import logistic_sgd_binomial

import mlp
import mlp_dropOut
import mlp_dropConnect

import convolutional_mlp
import con_mlp_dropConnect
import con_mlp_dropOut

logistic_cg.cg_optimization_mnist()
logistic_sgd.sgd_optimization_mnist()
logistic_sgd_gaussian.sgd_optimization_mnist()
logistic_sgd_binomial.sgd_optimization_mnist()

mlp.test_mlp()
# mlp_dropOut.test_mlp(p=0.8, n_hidden = 100)
mlp_dropOut.test_mlp()
mlp_dropConnect.test_mlp()

convolutional_mlp.evaluate_lenet5()
# con_mlp_dropConnect.evaluate_lenet5(p=0.8)
con_mlp_dropConnect.evaluate_lenet5()
con_mlp_dropOut.evaluate_lenet5()
