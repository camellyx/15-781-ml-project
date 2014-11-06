import sys

import numpy
import scipy

import convolutional_mlp
import logistic_cg
import logistic_sgd
import logistic_sgd_binomial
import logistic_sgd_binomial_error
import logistic_sgd_gaussian_gd
import mlp


# mnist
#logistic_sgd.sgd_optimization_mnist(n_epochs=10)

#logistic_cg.cg_optimization_mnist(n_epochs=10)

#logistic_sgd_binomial.sgd_optimization_mnist(n_epochs=10)

#logistic_sgd_binomial_error.sgd_optimization_mnist(n_epochs=10)

#mlp.test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=10,
#             dataset='mnist.pkl.gz', batch_size=20, n_hidden=50)

# cifar-10
#logistic_sgd.sgd_optimization_mnist(learning_rate=0.0005, n_epochs=50,
#                                    dataset='cifar-10-python.tar.gz',
#                                    batch_size=600)
#
#logistic_cg.cg_optimization_mnist(mnist_pkl_gz='cifar-10-python.tar.gz', n_epochs=10)
#
#logistic_sgd_binomial.sgd_optimization_mnist(dataset='cifar-10-python.tar.gz', n_epochs=10)
#
#logistic_sgd_binomial_error.sgd_optimization_mnist(dataset='cifar-10-python.tar.gz', n_epochs=10)
#
#logistic_sgd_gaussian_gd.sgd_optimization_mnist(dataset='cifar-10-python.tar.gz', n_epochs=10)
#
#mlp.test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=10,
#             dataset='cifar-10-python.tar.gz', batch_size=20, n_hidden=200)

# cifar-100
logistic_sgd.sgd_optimization_mnist(learning_rate=0.04, n_epochs=10,
                                    dataset='cifar-100-python.tar.gz',
                                    batch_size=600)

logistic_cg.cg_optimization_mnist(mnist_pkl_gz='cifar-100-python.tar.gz', n_epochs=10)

logistic_sgd_binomial.sgd_optimization_mnist(dataset='cifar-100-python.tar.gz', n_epochs=10)

logistic_sgd_binomial_error.sgd_optimization_mnist(dataset='cifar-100-python.tar.gz', n_epochs=10)

logistic_sgd_gaussian_gd.sgd_optimization_mnist(dataset='cifar-100-python.tar.gz', n_epochs=10)

mlp.test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=10,
             dataset='cifar-100-python.tar.gz', batch_size=20, n_hidden=50)
