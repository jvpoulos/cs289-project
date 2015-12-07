# Code adapted from https://github.com/Newmu/Theano-Tutorials

import theano
from theano import tensor as T
import numpy as np
execfile("load_data.py") # load training and validation sets
#from load import mnist

#x_train, x_test, y_train, y_test = mnist(onehot=True)

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
alpha = 2 # arbitrary scaling factor usually 2-10
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
updates = sgd(cost, params, gamma=10 ** -2)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# Test on validation set
n_batches = 128
max_epoch = 10

for i in range(max_epoch):
    for start, end in zip(range(0, len(x_train), n_batches), range(n_batches, len(x_train), n_batches)):
        cost = train(x_train[start:end], y_train[start:end])
    print 'epoch {}, accuracy {}, cost {}'.format(i, np.mean(np.argmax(y_test, axis=1) == predict(x_test)), cost)