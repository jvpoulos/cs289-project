import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse.linalg import svds
from scipy.stats import mode


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Imputer(object):
    def __init__(self):
        """
        Attributes
        ----------

        """

    def drop(self, x, missing_data_cond):
        # drop observations with missing values
        return x[np.sum(missing_data_cond(x), axis=1) == 0]


    def replace(self, x, missing_data_cond, in_place=False):
        """ Replace missing data with a random observation with data

        """
        if in_place:
            data = x
        else:
            data = np.copy(x)


        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:,col])
            val_ids = np.random.choice(np.where(nan_ids == False)[0],
                                       np.sum(nan_ids == True))
            data[nan_ids, col] = data[val_ids, col]
        return data


    def summarize(self, x, summary_func, missing_data_cond, in_place=False):
        """ Substitutes missing values with summary of each feature fector

        Parameters
        ----------
        summary_func : function
            Summarization function to be used for imputation
            (mean, median, mode, max, min...)
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # replace missing values with the mean or median of the feature
        for col in xrange(x.shape[1]):
            nan_ids = missing_data_cond(x[:,col])
            val, _ = summary_func(x[:,col])
            data[nan_ids, col] = val

        return data


    def one_hot(self, x, missing_data_cond, in_place=False):
        # create a one-hot row for each observation to mark whether or not
        # values are present

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        _, miss_cols = np.where(missing_data_cond(data))
        miss_cols_uniq = np.unique(miss_cols)

        for miss_col in miss_cols_uniq:
            uniq_vals, indices = np.unique(data[:,miss_col],
                                          return_inverse=True)

            data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                 dtype=int)[indices]))

        # remove categorical columns with missing data
        val_cols = [n for n in xrange(data.shape[1]) if n not in miss_cols_uniq]
        data = data[:, val_cols]
        return data



    def knn(self, x, k, summary_func, missing_data_cond, cols_cat = (1,3,4,5,6),
            in_place=False):
        """ Replace missing values with the mean or median of knn

        Parameters
        ----------
        k : int
            Number of nearest neighbors to be used

        """

        def row_col_from_condensed_idx(n_obs, i):
            b = 1 -2 * n_obs
            x = np.floor((-b - np.sqrt(b**2 - 8*i))/2)
            y = i + x*(b + x + 2)/2 + 1
            return (x, y)


        def condensed_idx_from_row_col(row, col, n_rows):
            return row*n_rows + col - row*(row+1)/2 - row - 1


        if in_place:
            data = x
        else:
            data = np.copy(x)

        imp = Imputer()

        # first transform features with categorical missing data into one hot
        data_complete = imp.one_hot(data, missing_data_cond)

        col_factors_labels = {}
        # replace categorical data with one hot rows
        for col_cat in cols_cat:
            factors, labels = pd.factorize(data_complete[:,col_cat])
            data_complete[:,col_cat] = factors
            col_factors_labels[col_cat] = (factors, labels)

        # convert data to int
        data_complete = data_complete.astype(int)

        # get indices of observations with nan
        miss_rows, miss_cols = np.where(missing_data_cond(data))
        n_obs = data_complete.shape[0]

        # compute distance matrix with nan values set to 0.0
        print 'Computing distance matrix'
        dist_mat = pdist(data_complete, metric='euclidean')

        print 'Substituting missing values'

        # substitute missing values with median of knn
        # this code must be optimized for speed!!!
        for j in xrange(len(miss_rows)):
            print 'Substituting {}-th of {} total'.format(j, len(miss_rows))
            miss_row_idx = miss_rows[j]
            # get indices of distances in condensed form
            ids_cond = [condensed_idx_from_row_col(miss_row_idx, idx, n_obs)
                        for idx in xrange(n_obs) if idx not in miss_rows]
            ids_cond = np.array(ids_cond, dtype=int)

            # compute k-nearest neighbors
            knn_indices_cond = ids_cond[np.argsort(dist_mat[ids_cond])[:k]]
            _, knn_indices = row_col_from_condensed_idx(n_obs, knn_indices_cond)

            # cols with missing data
            obs_nan_cols = np.where(missing_data_cond(x[miss_row_idx]))[0]

            knn_mean_vals, _ = mode(x[:,obs_nan_cols][knn_indices.astype(int)])

            data[miss_row_idx, obs_nan_cols] = knn_mean_vals.flatten()
        return data


    def predict(self, x, cat_cols, missing_data_cond, in_place=False):
        """ Uses random forest for predicting missing values

        Parameters
        ----------
        cat_cols : int tuple
            Index of columns that are categorical

        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # find rows and columns with missing data
        miss_rows, miss_cols = np.where(missing_data_cond(data))

        miss_cols_uniq = np.unique(miss_cols)
        valid_cols = [n for n in xrange(data.shape[1])
                      if n not in miss_cols_uniq]

        # factorize valid cols
        data_factorized = np.copy(data)

        # factorize categorical variables and store transformation
        factor_labels = {}
        for cat_col in cat_cols:
            factors, labels = pd.factorize(data[:, cat_col])
            factor_labels[cat_col] = labels
            data_factorized[:,cat_col] = factors

        # values are integers, convert accordingly
        data_factorized = data_factorized.astype(int)

        # update each column with missing features
        for miss_col in miss_cols_uniq:
            # edatatract valid observations given current column missing data
            valid_obs = [n for n in xrange(data.shape[0])
                         if data[n, miss_col] != '?']

            # prepare independent and dependent variables, valid obs only
            data_train = data_factorized[:, valid_cols][valid_obs]
            y_train = data_factorized[valid_obs, miss_col]

            # train random forest classifier
            rf_clf = RandomForestClassifier(n_estimators=100)
            rf_clf.fit(data_train, y_train)

            # given current feature, find obs with missing vals
            miss_obs_iddata = miss_rows[miss_cols == miss_col]

            # predict missing values
            y_hat = rf_clf.predict(data_factorized[:, valid_cols][miss_obs_iddata])

            # replace missing data with prediction
            data_factorized[miss_obs_iddata, miss_col] = y_hat

        # replace values on original data data
        for col in factor_labels.keys():
            data[:, col] = factor_labels[col][data_factorized[:, col]]

        return data


    def factor_analysis(self, x, cat_cols, missing_data_cond, threshold=0.9, in_place = False):
        """ Performs principal component analyze and replaces missing data with
        values obtained from the data procolected onto n principal components

        threshold : float
            Variance threshold that must be explained by eigen values.

        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        # get missing data indices
        nan_ids = missing_data_cond(x)

        # factorize valid cols
        data_factorized = np.copy(data)

        # factorize categorical variables and store transformation
        factor_labels = {}
        for cat_col in cat_cols:
            factors, labels = pd.factorize(x[:, cat_col])
            factor_labels[cat_col] = labels
            data_factorized[:,cat_col] = factors

        data_factorized = data_factorized.astype(float)

        # it's questionable whether high variance = high importance.
        u, s, vt = svds(data_factorized, data_factorized.shape[1] - 1,
                        which = 'LM')

        # find number of eigenvalues that explain 90% of variance
        sum_eigv = sum(s)
        n_pcomps = 1
        while sum(s[-n_pcomps:]) / sum_eigv < threshold:
            n_pcomps += 1

        # compute data procolected onto principal components space
        r = np.dot(u[:,-n_pcomps:],
                   np.dot(np.diag(s[-n_pcomps:]), vt[-n_pcomps:,]))

        data[nan_ids] = r[nan_ids].astype(int)

        return data


    def factorize_data(self, x, cols, in_place=False):
        """Replace column in cols with one-hot representation of cols

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data

        cols: tuple <int>
            Index of columns with categorical data

        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        factors_labels = {}
        for col in cols:
            factors, labels = pd.factorize(data[:,col])
            factors_labels[col] = (factors_labels)
            data[:,col] = factors

        return data, factor_labels

    def binarize_data(self, x, cols, one_minus_one=True, in_place=False):
        """Replace column in cols with one-hot representation of cols

        Parameters
        ----------
        x : np.ndarray
            Matrix with categorical data

        cols: tuple <int>
            Index of columns with categorical data

        Returns
        -------
        d : np.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = np.copy(x)

        for col in cols:
            uniq_vals, indices = np.unique(data[:,col],
                                          return_inverse=True)

            if one_minus_one:
                data = np.column_stack((data,
                    (np.eye(uniq_vals.shape[0], dtype=int)[indices] * 2) - 1))
            else:
                data = np.column_stack((data, np.eye(uniq_vals.shape[0],
                                                     dtype=int)[indices]))

        # remove columns with categorical variables
        val_cols = [n for n in xrange(data.shape[1]) if n not in cols]
        data = data[:, val_cols]
        return data