# Code adapted from https://github.com/Newmu/Theano-Tutorials
from itertools import product
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from sklearn.cross_validation import KFold
import time

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, gamma, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - gamma * g))
    return updates

def dropout(X, p=0.):
    srng = RandomStreams()
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return h, h2, py_x

# Cross-validation parameters
n_folds = 3

# Network topology
n_inputs = x_train.shape[1]
n_outputs = len(np.unique(y_train))

# Training parameters
alphas = np.arange(1, 11) # arbitrary scaling factor usually 2-10
gammas = np.power(10.0, np.arange(-1, -5, -1))
batch_sizes = np.power(2, np.arange(4,14))

# Dictionary to store results
results_dict = {}

params_matrix = np.array([x for x in product(alphas, gammas, batch_sizes)])
params_matrix = np.column_stack((params_matrix,
                                 np.zeros(params_matrix.shape[0]),
                                 np.zeros(params_matrix.shape[0]),
                                 np.zeros(params_matrix.shape[0])))

start = time.time()
for param_idx in xrange(params_matrix.shape[0]):
    alpha = params_matrix[param_idx, 0]
    gamma = params_matrix[param_idx, 1]
    batch_size = int(params_matrix[param_idx, 2])
    n_hidden = (x_train.shape[0] / n_folds)/(alpha*(n_inputs+n_outputs))
    
    # Initialize weights
    w_h = init_weights((n_inputs, n_hidden))
    w_h2 = init_weights((n_hidden, n_hidden))
    w_o = init_weights((n_hidden, n_outputs))
    
    # Initialize NN classifier
    X = T.fmatrix()
    Y = T.fmatrix()
    
    noise_h, noise_h2, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)
    h, h2, py_x = model(X, w_h, w_h2, w_o, 0., 0.)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    params = [w_h, w_h2, w_o]
    updates = RMSprop(cost, params, gamma=0.001)

    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
    
    # Test on validation set
    model_str = 'alpha {} gamma {} batchsize {}'.format(alpha,
                                                        gamma,
                                                        batch_size)
    max_epoch = 100
    print model_str

    kf = KFold(x_train.shape[0], n_folds=n_folds)
    error_rates = []
    test_costs = []
    
    fold = 1
    for train_idx, val_idx in kf:
        for i in range(max_epoch):
            for start, end in zip(range(0, len(x_train[train_idx]),
                                  batch_size),
                                  range(batch_size, len(x_train[train_idx]),
                                  batch_size)):
                test_cost = train(x_train[train_idx][start:end],
                                  y_train[train_idx][start:end])
                                        
            error_rate = 1 - np.mean(np.argmax(y_train[val_idx], axis=1) ==
                                               predict(x_train[val_idx]))
                                        
            print 'fold {}, epoch {}, error rate {}, cost {}'.format(fold, i+1,
                                                                     error_rate,
                                                                     test_cost)
        error_rates.append(error_rate)
        test_costs.append(test_cost)
        running_time.append(np.around((time.time() - start) / 60., 1))
        fold += 1
    
    params_matrix[param_idx, 3] = np.mean(error_rate)
    params_matrix[param_idx, 4] = np.mean(test_cost)
    params_matrix[param_idx, 5] = np.mean(running_time)
    print params_matrix[param_idx]

# Save params matrix to disk
params_matrix.dump('modern_net_results_' +str(imp_method) + '.np')