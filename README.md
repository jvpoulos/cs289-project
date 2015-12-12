# cs289-project
Methods of handling missing data for deep learning prediction model of earnings in [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult)

Given that we plan to use NNets for classification on the Adult dataset, we must handle missing data. This is not
necessary for other ML models such as random forest, decision trees, etc.

####Techniques for handling missing data
1. Basic Statistics : Replace the missing data with the mode, mean or median of the feature vector.
2. One-hot : Create an indicator variable to indicate whether or not the feature has missing data.
3. Nearest Neighbor Imputation : Recursively compute the K-Nearest Neighbors of the observation with missing data and
   assign the mean or median of the K-neighbors to the missing data.
4. Prediction: Use Random Forest to predict the feature with missing data.
6. Factor analysis : Perform eigen decomposition on the design matrix, project the design matrix onto the first two eigen
   vectors and replace the missing values by the values that might be given by
   the projected design matrix.

####Code folder
missing_data_imputation.py has the code for different imputation techniques
Params and result filenames can be generated with gen* files
Data can be generated with preprocess_missing_data.py
Plots can be generated with plot *
modern_net*.py and net*.py train the podels and predict error rate. 
