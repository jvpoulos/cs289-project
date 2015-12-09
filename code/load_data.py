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

# Binarize labels
labels = np.eye(2)[labels.astype(int)]
#labels_test = np.eye(2)[labels_test.astype(int)]

# Split to obtain train and test set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
