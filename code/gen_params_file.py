# saves params to a dictionary
import cPickle as pickle
import numpy as np

# Cross-validation and Neural Net parameters
params_dict = {}
params_dict['n_folds'] = 3
params_dict['alphas'] = (1,4,9)
params_dict['gammas'] = np.power(10.0, np.arange(-1, -3, -1))
params_dict['batch_sizes'] = (32, 512, 4096)
params_dict['max_epoch] = 50

# save pickled version
pickle.dump(params_dict, open('params_dict.pkl', 'wb'))