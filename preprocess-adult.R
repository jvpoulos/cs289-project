# Prepare adult dataset
# Uses code from http://scg.sdsu.edu/dataset-adult_r/

# Load data from UCI repository
adult <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                    sep=",",header=F,col.names=c("age", "type_employer", "fnlwgt", "education", 
                                                 "education_num","marital", "occupation", "relationship", "race","sex",
                                                 "capital_gain", "capital_loss", "hr_per_week","country", "income"),
                    fill=FALSE,strip.white=T)

# Missing to NA
is.na(adult) = adult=='?'
is.na(adult) = adult==' ?'

# Remove weight and education no. variables
adult <- subset(adult, select=-c(fnlwgt,education_num))

# Center and scale continuous variables
adult$age <- as.numeric(scale(adult$age, center = TRUE, scale = TRUE))
adult$capital_gain <- as.numeric(scale(adult$capital_gain, center = TRUE, scale = TRUE))
adult$capital_loss <- as.numeric(scale(adult$capital_loss, center = TRUE, scale = TRUE))
adult$hr_per_week <- as.numeric(scale(adult$hr_per_week, center = TRUE, scale = TRUE))

# Make response 0/1
adult$income = as.factor(ifelse(adult$income==adult$income[1],0,1))