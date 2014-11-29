import sys

import numpy
import scipy

import logistic_cg
import logistic_sgd
import logistic_sgd_gaussian
import logistic_sgd_binomial

import mlp
import mlp_dropOut
import mlp_dropConnect

import convolutional_mlp
import con_mlp_dropConnect
import con_mlp_dropOut

c100 = 'cifar-100-python.tar.gz'

sys.stdout = open('results/cifar-100_results/lcg.out','w')
logistic_cg.cg_optimization_mnist(mnist_pkl_gz = c100)

sys.stdout = open('results/cifar-100_results/lsgd.out','w')
logistic_sgd.sgd_optimization_mnist(dataset = c100)

sys.stdout = open('results/cifar-100_results/lsgd_gau.out','w')
logistic_sgd_gaussian.sgd_optimization_mnist(dataset = c100)

sys.stdout = open('results/cifar-100_results/lsgd_bin.out','w')
logistic_sgd_binomial.sgd_optimization_mnist(dataset = c100)

sys.stdout = open('results/cifar-100_results/mlp.out','w')
mlp.test_mlp(dataset = c100)

sys.stdout = open('results/cifar-100_results/mlpO.out','w')
# mlp_dropOut.test_mlp(p=0.8, n_hidden = 100)
mlp_dropOut.test_mlp(dataset = c100)

sys.stdout = open('results/cifar-100_results/mlpC.out','w')
mlp_dropConnect.test_mlp(dataset = c100)

sys.stdout = open('results/cifar-100_results/convo.out','w')
convolutional_mlp.evaluate_lenet5(dataset = c1000)

sys.stdout = open('results/cifar-100_results/convoC.out','w')
# con_mlp_dropConnect.evaluate_lenet5(p=0.8)
con_mlp_dropConnect.evaluate_lenet5(dataset = c100)

sys.stdout = open('results/cifar-100_results/convoO.out','w')
con_mlp_dropOut.evaluate_lenet5(dataset = c100)
