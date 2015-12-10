import numpy as np
import scipy as sp
import pandas as pd

from missing_data_imputation import Imputer

x = np.genfromtxt('../adult-dataset/adult-train-raw',
                      delimiter=', ', dtype=object)
# binarize labels
labels = (np.array(x[:,-1]) == '>50K').astype(int)
labels.dump('../adult-dataset/labels_bin.np')

# remove weight factor and label column
x = x[:,1:]
x = x[:,:-1]

# store valid information for generating data
rows, cols = np.where(x == '?')
full_obs = [i for i in xrange(x.shape[0]) if i not in rows]

# instantiate Imputer
imp = Imputer()
missing_data_cond = lambda x : x == '?'
cat_cols = (0, 2, 4, 5, 6, 7, 8, 12)
n_neighbors = 5

# drop missing variables, binarize and save complete observations and labels
data_drop = imp.drop(x, missing_data_cond)
data_drop_bin = imp.binarize_data(data_drop, cat_cols).astype(int)
data_drop_bin.dump('../adult-dataset/data_drop_bin.np')
labels[full_obs].dump('../adult-dataset/labels_drop_bin.np')

# replace missing values with random existing values
data_replace = imp.replace(x, missing_data_cond)
data_replace_bin = imp.binarize_data(data_replace, cat_cols).astype(int)
data_replace_bin.dump('../adult-dataset/data_replace_bin.np')

# replace missing values with feature mode
data_mode = imp.summarize(x, sp.stats.mode, missing_data_cond)
data_mode_bin = imp.binarize_data(data_mode, cat_cols).astype(int)
data_mode_bin.dump('../adult-dataset/data_mode_bin.np')

# repace categorical features with one hot row
data_one_hot = imp.binarize_data(x, cat_cols).astype(int)
data_one_hot.dump('../adult-dataset/data_onehot_bin.np')

# replace missing data with predictions
data_predicted = imp.predict(x, cat_cols, missing_data_cond)
data_predicted_bin = imp.binarize_data(data_predicted, cat_cols).astype(int)
data_predicted_bin.dump('../adult-dataset/data_predicted_bin.np')

# replace missing data with values obtained after factor analysis
data_facanal = imp.factor_analysis(x, cat_cols, missing_data_cond)
data_facanal_bin = imp.binarize_data(data_facanal, cat_cols).astype(int)
data_facanal_bin.dump('../adult-dataset/data_facanal_bin.np')

# replace missing data with knn
data_knn = imp.knn(x, n_neighbors, np.mean, missing_data_cond)
data_knn_bin = imp.binarize_data(data_knn, cat_cols).astype(int)
data_knn_bin.dump('../adult-dataset/data_knn_bin.np')