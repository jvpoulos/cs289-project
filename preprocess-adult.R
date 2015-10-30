# Prepare adult dataset
# Uses code from http://scg.sdsu.edu/dataset-adult_r/

# Load data from UCI repository
adult.train <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                          sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education", 
                                                       "education_num","marital", "occupation", "relationship", "race","sex",
                                                       "capital_gain", "capital_loss", "hr_per_week","country", "income"),
                          fill=FALSE,strip.white=T)


adult.test <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                          sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education", 
                                                       "education_num","marital", "occupation", "relationship", "race","sex",
                                                       "capital_gain", "capital_loss", "hr_per_week","country", "income"),
                          fill=FALSE,strip.white=T,skip=1)

PreProcessAdult <- function(data){
  # Preprocess UCI Adult data. 
  #
  # Args:
  #   data: Dataframe of features and response.
  #
  # Returns:
  #   Dataframe of preprocessed data. 
  
  # Remove technical variables
  data <- subset(data, select=-c(fnlwgt,education_num))
  
  # ? to NA
  is.na(data) = data=='?'
  is.na(data) = data==' ?'
  
  # Center and scale continuous variables
  data$age <- as.numeric(scale(data$age, center = TRUE, scale = TRUE))
  data$capital_gain <- as.numeric(scale(data$capital_gain, center = TRUE, scale = TRUE))
  data$capital_loss <- as.numeric(scale(data$capital_loss, center = TRUE, scale = TRUE))
  data$hr_per_week <- as.numeric(scale(data$hr_per_week, center = TRUE, scale = TRUE))
  
  # Make response 0/1
  data$income = as.factor(ifelse(data$income==data$income[1],0,1))
  
  return(data)
}

# Preprocess and export data

adult.train <- PreProcessAdult(adult.train)

write.table(adult.train,"adult-train.csv")

adult.test <- PreProcessAdult(adult.test)

write.table(adult.train,"adult-test.csv")
