import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import svds

class MDH(Object):
    def __init__(self, technique, summary_func, k=5, pca_threshold=0.9):
        """
        Attributes
        ----------
        technique : string
            String representation of the imputation technique to be used

        summary_func : string
            String representation of summary_func of imputation (mean, median)
        """

        self.technique = technique

        if summary_func == 'mean':
            self.summary_func = np.namnean
        elif summary_func == 'median':
            self.summary_func = np.nanmedian
        self.k = k
        self.pca_threshold = pca_threshold

    def impute(self, x):
        """
        Parameters
        ----------
        x : np.ndarary
            Design matrix where rows are observations and columns are features

        Returns
        -------
        x : np.ndarray
            New design matrix with missing data imputed using the respective
            technique and summary_func
        """

        if self.technique == 'drop':
            # drop observations with missing values
            return x[np.sum(np.isnan(x), axis=1) == 0]
        elif self.technique == 'case substitution':
            # replace missing values with a random observation that has a value
            for col in xrange(x.shape[1]):
                nan_ids = np.isnan(x[:,col])
                val_ids = np.random.choice(np.where(nan_ids == False)[0],
                                           np.sum(nan_indices == True))
                x[nan_ids, col] = x[val_ids, col]
            return x
        elif self.technique == 'imputation':
            # replace missing values with the mean or median of the feature
            for col in xrange(x.shape[1]):
                nan_ids = np.isnan(x[:,col])
                val_median = self.summary_func(x[:,col])
                x[nan_ids, col] = val_median
            return x
        elif self.technique == 'one-hot':
            # create a one-hot row for each observation to mark whether or not
            # values are present
            return np.column_stack((x, np.isnan(x)))
        elif self.technique == 'one-hot knn':
            # computes knn by comparing existing values and one hot row
        elif self.technique == 'knn':
            # replace missing values with the mean or median of knn

            # get indices of observations with nan
            nan_obs_ids = [idx for idx in xrange(x.shape[0])
                           if any(np.isnan(x[idx]))]

            # there are no nans, return the unmodified design matrix
            if len(nan_obs_ids) == 0:
                return x

            # compute distance matrix with nan values set to 0.0
            dist_mat = euclidean_distances(np.nan_to_num(x))

            # substitute missing values with median of knn
            for obs_idx in nan_obs_ids:
                # compute k-nearest neighbors
                knn_indices = np.argsort(dist_mat[obs_idx])[1:k+1]
                knn_mean_vals = self.summary_func((x[knn_indices], axis=0))
                obs_nan_ids = np.where(np.isnan(x[obs_idx]))[0]
                x[obs_idx, obs_nan_ids] = knn_mean_vals[obs_nan_ids]
        elif self.technique == 'prediction':
        elif self.technique == 'factor analysis':
            u, s, vt = svds(x, d, which = 'LM')
            r = np.dot(u, np.dot(np.diag(s), vt))
            mse_pcs[d] = np.mean((r - train_data_clean) ** 2)
        else:
            print 'Technique {} is not supported'.format(technique)
            return None