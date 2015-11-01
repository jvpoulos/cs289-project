# cs289-project
Methods of handling missing data for deep learning prediction model of earnings in [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult)

Given that we plan to use NNets for classification on the Adult dataset, we must handle missing data. This is not
necessary for other ML models such as random forest, decision trees, etc.

####Techniques for handling missing data
1. Basic Statistics : Replace the missing data with the mean or median of the feature vector.
2. One-hot : Create an indicator variable to indicate whether or not the feature has missing data.
3. Nearest Neighbor Imputation : Recursively compute the K-Nearest Neighbors of the observation with missing data and
   assign the mean or median of the K-neighbors to the missing data.
4. Regression : Recursively train a regression model to predict the feature with missing
   data.
5. Bagging : Use one bag tree model for each predictor based on all other
   predictors.
6. Factor analysis : Perform eigen decomposition on the design matrix, project the design matrix onto the first two eigen
   vectors and replace the missing values by the values that might be given by
   the projected design matrix.
7. Find other features with distribution similar to the feature containing missing data and use this information (e.g. correlation) to fill in in the missing data. But then, if two features are highly correlated, it might be better to remove one of them.

####Thoughts
Filling in the wrong values might add bias to the prediction
Missing completely at random, missing at random, missing not at random?
