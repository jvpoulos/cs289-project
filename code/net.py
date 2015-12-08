# Code adapted from https://github.com/Newmu/Theano-Tutorials
from itertools import product
import theano
#import pydot
from theano import tensor as T
import numpy as np
execfile("load_data.py") # load training and validation sets

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, gamma):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * gamma])
    return updates

def model(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))
    pyx = T.nnet.softmax(T.dot(h, w_o))
    return pyx

# Network topology
n_inputs = x_train.shape[1]
n_outputs = len(np.unique(y_train))

# training parameters
alphas = np.arange(1, 11) # arbitrary scaling factor usually 2-10
gammas = np.power(10.0, np.arange(-1, -5, -1))
batch_sizes = np.power(2, np.arange(4,14))

# dictionary to store results
results_dict = {}

params_matrix = np.array([x for x in product(alphas, gammas, batch_sizes)])
params_matrix = np.column_stack((params_matrix,
                                 np.zeros(params_matrix.shape[0]),
                                 np.zeros(params_matrix.shape[0])))

for param_idx in xrange(params_matrix.shape[0]):
    alpha = params_matrix[param_idx, 0]
    gamma = params_matrix[param_idx, 1]
    batch_size = int(params_matrix[param_idx, 2])
    n_hidden = (x_train.shape[0])/(alpha*(n_inputs+n_outputs))

    # Initialize weights
    w_h = init_weights((n_inputs, n_hidden))
    w_o = init_weights((n_hidden, n_outputs))

    # Initialize NN classifier
    X = T.fmatrix()
    Y = T.fmatrix()

    py_x = model(X, w_h, w_o)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    params = [w_h, w_o]

    updates = sgd(cost, params, gamma=gamma)

    train = theano.function(inputs=[X, Y],
                            outputs=cost,
                            updates=updates,
                            allow_input_downcast=True)

    predict = theano.function(inputs=[X],
                              outputs=y_x,
                              allow_input_downcast=True)

    # Draw graph
    # theano.printing.pydotprint(predict,
    #                            outfile="nnet_predict_"+model_str+".png",
    #                            var_with_name_simple=True)

    # Test on validation set
    model_str = 'alpha {} gamma {} batchsize {}'.format(alpha,
                                                        gamma,
                                                        batch_size)
    max_epoch = 2
    print model_str
    for i in range(max_epoch):
        for start, end in zip(range(0, len(x_train), batch_size),
                              range(batch_size, len(x_train), batch_size)):
            test_cost = train(x_train[start:end], y_train[start:end])
        error_rate = 1 - np.mean(np.argmax(y_test, axis=1) == predict(x_test))
        print 'epoch {}, error rate {}, cost {}'.format(i,
                                                        error_rate,
                                                        test_cost)
    params_matrix[param_idx, 3] = error_rate
    params_matrix[param_idx, 4] = test_cost
    print params_matrix[param_idx]

# save params matrix to disk
params_matrix.dump('net_results.np')
