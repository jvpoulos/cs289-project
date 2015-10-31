# Prepare adult dataset

# Load data from UCI repository
adult.train <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                          sep=",",header=F,col.names=c("age", "type.employer", "fnlwgt", "education", 
                                                       "education.num","marital", "occupation", "relationship", "race","sex",
                                                       "capital.gain", "capital.loss", "hr.per.week","country", "income"),
                          fill=FALSE,strip.white=T)


adult.test <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                          sep=",",header=F,col.names=c("age", "type.employer", "fnlwgt", "education", 
                                                       "education.num","marital", "occupation", "relationship", "race","sex",
                                                       "capital.gain", "capital.loss", "hr.per.week","country", "income"),
                          fill=FALSE,strip.white=T,skip=1)

PreProcessAdult <- function(data){
  # Preprocess UCI Adult data. 
  #
  # Args:
  #   data: Dataframe of features and response.
  #
  # Returns:
  #   Dataframe of preprocessed data. 
  
  # ? to NA
  is.na(data) = data=='?'
  is.na(data) = data==' ?'
  
  # Center and scale continuous variables
  data$age <- as.numeric(scale(data$age, center = TRUE, scale = TRUE))
  data$capital.gain <- as.numeric(scale(data$capital.gain, center = TRUE, scale = TRUE))
  data$capital.loss <- as.numeric(scale(data$capital.loss, center = TRUE, scale = TRUE))
  data$hr.per.week <- as.numeric(scale(data$hr.per.week, center = TRUE, scale = TRUE))
  data$fnlwgt <- as.numeric(scale(data$fnlwgt, center = TRUE, scale = TRUE))
  data$education.num <- as.numeric(scale(data$education.num, center = TRUE, scale = TRUE))
  
  # Make response 0/1
  data$income = as.factor(ifelse(data$income==data$income[1],0,1))
  
  return(data)
}

# Preprocess and export data

adult.train <- PreProcessAdult(adult.train)

write.table(adult.train,"adult-train.csv")

adult.test <- PreProcessAdult(adult.test)

write.table(adult.train,"adult-test.csv")
