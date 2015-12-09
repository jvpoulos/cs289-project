import numpy as np
from sklearn.cross_validation import train_test_split

# Load train data
features = np.genfromtxt('../adult-dataset/adult-train-features-median.csv',
                         delimiter=' ',
                         skip_header=1) # train, load from csv without headers
features = features[:,1:] # remove index column

labels = np.genfromtxt('../adult-dataset/adult-train-labels.csv',
                       delimiter=' ',
                       skip_header=1) # need to remove quotes

labels= labels[:,1:][:,0] # remove index column

# Split to obtain train and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33)

# Binarize labels
labels_train = np.eye(2)[labels_train.astype(int)]
labels_test = np.eye(2)[labels_test.astype(int)]