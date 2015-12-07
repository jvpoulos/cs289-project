# Code adapted from https://github.com/Newmu/Theano-Tutorials
import theano
import pydot
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
alphas = np.arange(1, 11) # arbitrary scaling factor usually 2-10

# dictionary to store results
results_dict = {}

for alpha in alphas:
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
    gammas = np.power(10.0, np.arange(-1, -5, -1))

    for gamma in gammas:
        batch_sizes = np.power(2, np.arange(4,14))
        for batch_size in batch_sizes:
            model_str = 'alpha {} gamma {} batchsize {}'.format(alpha,
                                                                gamma,
                                                                batch_size)
            updates = sgd(cost, params, gamma=gamma)

            train = theano.function(inputs=[X, Y],
                                    outputs=cost,
                                    updates=updates,
                                    allow_input_downcast=True)

            # Draw graph
            theano.printing.pydotprint(train,
                                       outfile="nnet_train.png",
                                       var_with_name_simple=True)

            predict = theano.function(inputs=[X],
                                      outputs=y_x,
                                      allow_input_downcast=True)

            # Test on validation set
            max_epoch = 10
            print model_str
            for i in range(max_epoch):
                for start, end in zip(range(0, len(x_train), batch_size),
                                      range(batch_size, len(x_train), batch_size)):
                    test_cost = train(x_train[start:end], y_train[start:end])
                error_rate = 1 - np.mean(np.argmax(y_test, axis=1) == predict(x_test))
                print 'epoch {}, error rate {}, cost {}'.format(i,
                                                                error_rate,
                                                                test_cost)
            results_dict[model_str] = (error_rate, cost)

# Test on validation set
n_batches = 128
max_epoch = 1000

for i in range(max_epoch):
    for start, end in zip(range(0, len(x_train), n_batches), range(n_batches, len(x_train), n_batches)):
        cost = train(x_train[start:end], y_train[start:end])
    print 'epoch {}, error {}, cost {}'.format(i, (1-np.mean(np.argmax(y_test, axis=1) == predict(x_test))), cost)
>>>>>>> dc1543396d6a48e2e60a968e022510af12ef4278
