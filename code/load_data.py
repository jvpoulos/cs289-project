import numpy as np
from sklearn.cross_validation import train_test_split

# Load training data
x_train = np.genfromtxt('../adult-dataset/adult-train-features-median.csv',
                         delimiter=' ',
                         skip_header=1) # train, load from csv without headers
x_train = x_train[:,1:] # remove index column

y_train = np.genfromtxt('../adult-dataset/adult-train-labels.csv',
                       delimiter=' ',
                       skip_header=1) # need to remove quotes

y_train = y_train[:,1:][:,0] # remove index column

# Binarize labels
y_train = np.eye(2)[y_train.astype(int)]