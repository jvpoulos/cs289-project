# Create plots for exploratory data analysis 

# Libraries
require(ggplot2)
#require(GGally)
require(VIM)

# Load data from UCI repository
adult.train <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                          sep=",",header=F,col.names=c("age", "workclass", "fnlwgt", "education", 
                                                       "education.num","marital.status", "occupation", "relationship", "race","sex",
                                                       "capital.gain", "capital.loss", "hours.per.week","native.country", "label"),
                          fill=FALSE,strip.white=T)

# Drop education string and label
adult.train <- subset(adult.train, select=-c(education, label))

# Convert ? to NA
is.na(adult.train) <- adult.train=='?'

missing <- c("workclass","occupation","native.country")

for(x in missing) {
  adult.train[x] <- droplevels(adult.train[x])
}

# Plot the amount of missing values in each variable 
pdf("missing-values-patterns.pdf", width=11.69, height=8.27)
aggr(adult.train, col=c('navyblue','red'), 
     numbers=TRUE, 
     sortVars=TRUE, 
     labels=c("Age","Work class", "Sample weight", "Education", "Marital status", "Occupation", "Relationship", "Race", "Sex", "Capital gain", "Capital loss", "Hours/week", "Native country"), 
     cex.lab=.7,
     cex.axis=.7,
     cex.numbers=.7,
     gap=3, 
     ylab=c("Proportion of missing values per feature",
            "Proportion of missing values per feature combination"))
dev.off() 
