# Preprocess adult dataset

# Libraries
require(devtools) 
devtools::install_github("jeffwong/imputation") # import imputation methods
require(imputation)
require(caret)
require(matrixStats)

# notes: use median and mean for KNN

# Load data from UCI repository
adult.train <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                          sep=",",header=F,col.names=c("age", "workclass", "fnlwgt", "education", 
                                                       "education.num","marital.status", "occupation", "relationship", "race","sex",
                                                       "capital.gain", "capital.loss", "hours.per.week","native.country", "label"),
                          fill=FALSE,strip.white=T)


adult.test <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                         sep=",",header=F,col.names=c("age", "workclass", "fnlwgt", "education", 
                                                      "education.num","marital.status", "occupation", "relationship", "race","sex",
                                                      "capital.gain", "capital.loss", "hours.per.week","native.country", "label"),
                         fill=FALSE,strip.white=T,skip=1)

# Preprocess data

PreProcessAdult <- function(train,test,imp.method="none",scale.method=2){
  # Preprocess UCI Adult data. 
  #
  # Args:
  #   train: Dataframe of training features and response.
  #   test: Dataframe of test features and response. 
  #   imp.method: Method for data imputation.
  #     Options are: "mean", "median", "gbm", "knn.mean", "knn.median", "lm", "svd", "svt", or "none". Default is "none".
  #   scale.method:  1: Mean 0 and standard deviation 1; or 2: Midrange 0 and range 2 (i.e., minimum -1 and maximum 1). Default is 2.
  #
  # Returns:
  #   List containing labels and preprocessed features for train and test sets.
  
  # Make labels binary
  train$label <- as.factor(ifelse(train$label==train$label[1],0,1))
  test$label <- as.factor(ifelse(test$label==test$label[1],0,1))
  
  # Drop education string
  train <- subset(train, select=-c(education))
  test <- subset(test, select=-c(education))
  
  # Convert ? to NA
  is.na(train) <- train=='?'
  is.na(test) <- test=='?'
  
  missing <- c("workclass","occupation","native.country")
  
  for(x in missing) {
    train[x] <- droplevels(train[x])
    test[x] <- droplevels(test[x])
  }
  
  # Replace dashes with periods in factor labels
  ReplaceDashes <- function(feature) {
    feature <- gsub("-",".",feature)
    return(as.factor(feature))
  }
  
  categorical <- c("workclass", "marital.status", "occupation","relationship","race","native.country")
  
  for(x in categorical) {
    train[,x] <- ReplaceDashes(train[,x])
    test[,x] <- ReplaceDashes(test[,x])
  }
  
  # Create dummy variables for factor variables
  train.dummies <- dummyVars(label ~ ., data = train)
  train.bin <- predict(train.dummies, newdata = train)
  
  test.dummies <- dummyVars(label ~ ., data = test)
  test.bin <- predict(test.dummies, newdata = test)
  
  # Impute missing values
  if(imp.method=="median"){
    median.train <- sapply(colnames(train.bin), function(x){
      median(train.bin[,x], na.rm=TRUE)
    })
    train.impute <- train.bin
    for(x in 1:ncol(train.bin)){
      train.bin[,x][is.na(train.bin[,x])] <- median.train[x]
    }
    test.impute <- test.bin
    for(x in 1:ncol(test.bin)){
      test.bin[,x][is.na(test.bin[,x])] <- median.train[x] # use train median to impute test set
    }
  }

  if(imp.method=="mean"){
    mean.train <- sapply(colnames(train.bin), function(x){
      mean(train.bin[,x], na.rm=TRUE)
    })
    train.impute <- train.bin
    for(x in 1:ncol(train.bin)){
      train.bin[,x][is.na(train.bin[,x])] <- mean.train[x]
    }
    test.impute <- test.bin
    for(x in 1:ncol(test.bin)){
      test.bin[,x][is.na(test.bin[,x])] <- mean.train[x] # use train mean to impute test set
    }
  }
  
  if(imp.method=="gbm"){
    train.impute <- gbmImpute(train.bin)
    test.impute <- gbmImpute(test.bin)
  }
  if(imp.method=="knn.mean"){
    train.impute <- kNNImpute(train.bin)
    test.impute <- kNNImpute(test.bin)
  }
  if(imp.method=="knn.median"){
    impute.fn <- function(scores, distances, raw_dist) {
      knn.values <- scores[c(as.integer(names(distances)))]
      knn.weights <- 1 - (distances / max(raw_dist))
      return(matrixStats::weightedMedian(knn.values, knn.weights))
    }
    
    train.impute <- kNNImpute(train.bin, impute.fn)
    test.impute <- kNNImpute(test.bin, impute.fn)
  }
  if(imp.method=="lm"){
    train.impute <- lmImpute(train.bin)
    test.impute <- lmImpute(test.bin)
  }
  if(imp.method=="svd"){
    train.impute <- SVDImpute(train.bin, k=3)
    test.impute <- SVDImpute(test.bin, k=3)
  }
  if(imp.method=="svt"){
    train.impute <- SVTImpute(train.bin)
    test.impute <- SVTImpute(test.bin)
  }
  else{
    train.impute <- train.bin
    test.impute <- test.bin
  }
  
  continuous <- c("age","capital.gain","capital.loss","hours.per.week","fnlwgt","education.num")
  
  # Standardize features
  if(scale.method==1){
    # Center and scale continuous features
    center.scale <- preProcess(train.impute[,continuous], method = c("center","scale")) 
    train.impute[,continuous] <- predict(center.scale, newdata=train.impute[,continuous]) 
    test.impute[,continuous] <- predict(center.scale, newdata=test.impute[,continuous])  # scale test set with training set mean and sd
  }
  
  if(scale.method==1){
    # Standardize X to mean 0 and standard deviation 1:
    mean.train <- sapply(colnames(train.impute), function(x){
      mean(train.impute[,x],na.rm=TRUE)
    })  
    
    sd.train <- sapply(colnames(train.impute), function(x){ 
      sd(train.impute[,x],na.rm=TRUE)
    }) 
    
    for(x in 1:ncol(train.impute)){
      train.impute[,x] <- (train.impute[,x] - mean.train[x])/(sd.train[x])
    }
    
    for(x in 1:ncol(test.impute)){
      test.impute[,x] <- (test.impute[,x] - mean.train[x])/(sd.train[x]) # use train set mean and sd
    }
  }
  
  if(scale.method==2){
    # Midrange 0 and range 2 (-1 to 1) for ALL features
    midrange.train <- sapply(colnames(train.impute), function(x){
      (max(train.impute[,x],na.rm=TRUE) + min(train.impute[,x],na.rm=TRUE))/2
    })  
    
    range.train <- sapply(colnames(train.impute), function(x){ 
      max(train.impute[,x],na.rm=TRUE) - min(train.impute[,x],na.rm=TRUE)
    }) 
    
    for(x in 1:ncol(train.impute)){
      train.impute[,x] <- (train.impute[,x] - midrange.train[x])/(range.train[x]/2)
    }
    
    for(x in 1:ncol(test.impute)){
      test.impute[,x] <- (test.impute[,x] - midrange.train[x])/(range.train[x]/2) # use train set midrange and range values
    }
  }
  
  return(list("train.features"=subset(train.impute, select=-c(native.country.Holand.Netherlands)), # make sure no. features same
              "train.labels"=train$label,
              "test.features"=test.impute,
              "test.labels"=test$label))
}
## Preprocess and export data

# No imputation
adult.pre <- PreProcessAdult(adult.train, adult.test, imp.method="none") 

write.table(adult.pre[["train.labels"]],"adult-train-labels.csv", quote=FALSE) 
write.table(adult.pre[["test.labels"]],"adult-test-labels.csv",quote=FALSE)

write.table(adult.pre[["train.features"]],"adult-train-features-none.csv")
write.table(adult.pre[["test.features"]],"adult-test-features-none.csv")

# Median imputation
adult.pre.median <- PreProcessAdult(adult.train, adult.test, imp.method="median") 

write.table(adult.pre.median[["train.features"]],"adult-train-features-median.csv")
write.table(adult.pre.median[["test.features"]],"adult-test-features-median.csv")

# Mean imputation
adult.pre.mean <- PreProcessAdult(adult.train, adult.test, imp.method="mean") 

write.table(adult.pre.mean[["train.features"]],"adult-train-features-mean.csv")
write.table(adult.pre.mean[["test.features"]],"adult-test-features-mean.csv")
