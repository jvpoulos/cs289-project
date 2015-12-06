import numpy as np
from sklearn.cross_validation import train_test_split

# Load train data
# train, load from csv without headers
features_train = np.genfromtxt('../adult-dataset/adult-train-features-median.csv', delimiter=' ', skip_header=1)
features_train = features_train[:,1:] # remove index column

labels_train = np.genfromtxt('../adult-dataset/adult-train-labels.csv', delimiter=' ', skip_header=1) # need to remove quotes
labels_train = labels_train[:,1:][:,0] # remove index column

# split to obtain train and test set
x_train, x_test, y_train, y_test = train_test_split(features_train, labels_train, test_size=0.33)