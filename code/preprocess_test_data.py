import numpy as np
import cPickle as pickle
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from missing_data_imputation import Imputer

#declare csv headers
x = np.genfromtxt('../adult-dataset/adult-test-raw',
                      delimiter=', ', dtype=object)

# binarize labels
labels = (np.array(x[:,-1]) == '>50K').astype(int)
labels.dump('../adult-dataset/labels_test_bin.np')
# dump binarized labels
(np.eye(2)[labels.astype(int)]).astype(int).dump('../adult-dataset/labels_test_onehot.np')

# remove label column
x = x[:,:-1]

# remove redundant education-number feature
relevant_cols = [i for i in xrange(x.shape[1]) if i != 4]
x = x[:, relevant_cols]

# store valid information for generating data
rows, cols = np.where(x == '?')
full_obs = [i for i in xrange(x.shape[0]) if i not in rows]
labels[full_obs].dump('../adult-dataset/labels_test_drop_bin.np')
# dump binarized labels full obs
(np.eye(2)[labels[full_obs].astype(int)]).astype(int).dump('../adult-dataset/labels_test_drop_bin_onehot.np')

# enumerate parameters and instantiate Imputer
imp = Imputer()
missing_data_cond = lambda x : x == '?'
cat_cols = (1, 3, 4, 5, 6, 7, 8, 12)
n_neighbors = 5

# drop missing variables, binarize and save complete observations and labels
print 'imputing with drop'
data_drop = imp.drop(x, missing_data_cond)
np.savetxt("../adult-dataset/data_test_drop.csv", data_drop, delimiter=",", fmt="%s")
data_drop_bin = imp.binarize_data(data_drop, cat_cols).astype(float)
data_drop_bin.dump('../adult-dataset/data_test_drop_bin.np')
#load scaled
scaler = pickle.load(open('scaler_drop.pkl', 'rb'))
data_drop_bin_scaled = scaler.transform(data_drop_bin)
data_drop_bin_scaled.dump('../adult-dataset/data_test_drop_bin_scaled.np')
del data_drop
del data_drop_bin
del data_drop_bin_scaled


# replace missing values with random existing values
print 'imputing with replace'
data_replace = imp.replace(x, missing_data_cond)
np.savetxt("../adult-dataset/data_replace.csv", data_replace, delimiter=",", fmt="%s")
data_replace_bin = imp.binarize_data(data_replace, cat_cols).astype(float)
data_replace_bin.dump('../adult-dataset/data_replace_bin.np')
scaler = StandardScaler().fit(data_replace_bin)
data_replace_bin_scaled = scaler.transform(data_replace_bin)
data_replace_bin_scaled.dump('../adult-dataset/data_replace_bin_scaled.np')
del data_replace
del data_replace_bin
del data_replace_bin_scaled


# replace missing values with feature mode
print 'imputing with mode'
data_mode = imp.summarize(x, mode, missing_data_cond)
np.savetxt("../adult-dataset/data_test_mode.csv", data_mode, delimiter=",", fmt="%s")
data_mode_bin = imp.binarize_data(data_mode, cat_cols).astype(float)
data_mode_bin.dump('../adult-dataset/data_test_mode_bin.np')
scaler = pickle.load(open('scaler_mode.pkl', 'rb'))
data_mode_bin_scaled = scaler.transform(data_mode_bin)
data_mode_bin_scaled.dump('../adult-dataset/data_test_mode_bin_scaled.np')
del data_mode
del data_mode_bin
del data_mode_bin_scaled


# repace categorical features with one hot row
print 'imputing with onehot'
data_onehot = imp.binarize_data(x, cat_cols).astype(float)
data_onehot.dump('../adult-dataset/data_onehot_bin.np')
scaler = StandardScaler().fit(data_onehot)
data_onehot_scaled = scaler.transform(data_onehot)
data_onehot_scaled.dump('../adult-dataset/data_onehot_bin_scaled.np')
del data_onehot
del data_onehot_scaled


# replace missing data with predictions
print 'imputing with predicted'
data_predicted = imp.predict(x, cat_cols, missing_data_cond)
np.savetxt("../adult-dataset/data_test_predicted.csv", data_predicted, delimiter=",", fmt="%s")
data_predicted_bin = imp.binarize_data(data_predicted, cat_cols).astype(float)
data_predicted_bin.dump('../adult-dataset/data_test_predicted_bin.np')
scaler = pickle.load(open('scaler_predicted.pkl', 'rb'))
data_predicted_bin_scaled = scaler.transform(data_predicted_bin)
data_predicted_bin_scaled.dump('../adult-dataset/data_test_predicted_bin_scaled.np')
del data_predicted
del data_predicted_bin
del data_predicted_bin_scaled


# replace missing data with values obtained after factor analysis
print 'imputing with factor analysis'
data_facanal = imp.factor_analysis(x, cat_cols, missing_data_cond)
np.savetxt("../adult-dataset/data_facanal.csv", data_facanal, delimiter=",", fmt="%s")
data_facanal_bin = imp.binarize_data(data_facanal, cat_cols).astype(float)
data_facanal_bin.dump('../adult-dataset/data_facanal_bin.np')
scaler = StandardScaler().fit(data_facanal_bin)
data_facanal_bin_scaled = scaler.transform(data_facanal_bin)
data_facanal_bin_scaled.dump('../adult-dataset/data_facanal_bin_scaled.np')
del data_facanal
del data_facanal_bin
del data_facanal_bin_scaled

# replace missing data with knn
#data_knn = imp.knn(x, n_neighbors, np.mean, missing_data_cond)
#data_knn_bin = imp.binarize_data(data_knn, cat_cols).astype(float)
#data_knn_bin.dump('../adult-dataset/data_knn_bin.np')
#scaler = StandardScaler().fit(data_knn_bin)
#data_knn_bin_scaled = scaler.transform(data_knn_bin)
#data_knn_bin_scaled.dump('../adult-dataset/data_knn_bin_scaled.np')