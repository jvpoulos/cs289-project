# Code adapted from https://github.com/Newmu/Theano-Tutorials
import sys, time
from ntpath import basename
from os.path import splitext
from itertools import product
import cPickle as pickle
import theano
from theano import tensor as T
import numpy as np
from sklearn.cross_validation import KFold


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

# get filename for saving results
filename =  splitext(basename(sys.argv[0]))[0]
print 'Executing {}'.format(filename)

# Load training and test sets
x_train = np.load('../adult-dataset/data_mode_bin_scaled.np')
y_train = np.load('../adult-dataset/labels_onehot.np')
x_test = np.load('../adult-dataset/data_test_mode_bin_scaled.np')
y_test = np.load('../adult-dataset/labels_test_onehot.np')


# Network topology
n_inputs = x_train.shape[1]
n_outputs = len(np.unique(y_train))

# Cross-validation and Neural Net parameters
#params_dict = pickle.load(open('params_dict.pkl', 'rb'))
alphas = (4,)
gammas = (0.1,)
batch_sizes = (32,)
max_epoch = 50

# Dictionary to store results
results_dict = {}

params_matrix = np.array([x for x in product(alphas, gammas, batch_sizes)])
params_matrix = np.column_stack((params_matrix,
                                 np.zeros(params_matrix.shape[0]),
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


    # Test on validation set
    model_str = 'alpha {} gamma {} batch size {}'.format(alpha,
                                                        gamma,
                                                        batch_size)
    print model_str

    error_rates = []
    test_costs = []
    running_time = []

    start_time = time.time()
    for i in range(max_epoch):
        for start, end in zip(range(0, len(x_train),
                              batch_size),
                              range(batch_size, len(x_train),
                              batch_size)):
            test_cost = train(x_train[start:end],
                              y_train[start:end])

        error_rate = 1 - np.mean(np.argmax(y_train, axis=1) == predict(x_train))

        print 'epoch {}, error rate {}, cost {}'.format(i+1,
                                                        error_rate,
                                                        test_cost)
    error_rates.append(error_rate)
    test_costs.append(test_cost)
    running_time.append(np.around((time.time() - start_time) / 60., 1))

    params_matrix[param_idx, 3] = np.mean(error_rate)
    params_matrix[param_idx, 4] = np.mean(test_cost)
    params_matrix[param_idx, 5] = np.mean(running_time)

    print 'alpha {} gamma {} batchsize {} error rate {} test cost {} running time {}'.format(params_matrix[param_idx,0],
        params_matrix[param_idx,1],
        params_matrix[param_idx,2],
        params_matrix[param_idx,3],
        params_matrix[param_idx,4],
        params_matrix[param_idx,5])

    # predict on test data
    error_rate_test = 1 - np.mean(np.argmax(y_test, axis=1) == predict(x_test))
    print 'Test Error rate : {}'.format(error_rate_test)


# test error rate with 50 epochs : 0.1894232541
# test error rate with 10 epochs : 0.176279098335
# test error rate with 5 epoch : 0.173146612616
# test error rate with 1 epoch : 0.168417173392

# Save params matrix to disk
params_matrix.dump('{}_results.np'.format(filename))
