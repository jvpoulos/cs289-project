# Create plots for exploratory data analysis 

# Libraries
require(VIM)
require(ggplot2)
require(gridExtra)

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
pdf("proportion-missing.pdf", width=11.69, height=8.27)
aggr(adult.train, col=c('navyblue','red'), 
     numbers=TRUE, 
     sortVars=TRUE, 
     labels=c("Age","Work class", "Sample weight", "Education", "Marital status", "Occupation", "Relationship", "Race", "Sex", "Capital gain", "Capital loss", "Hours/week", "Native country"), 
     cex.lab=.6,
     cex.axis=.6,
     cex.numbers=.7,
     gap=3, 
     ylab=c("Proportion of missing values per feature",
            "Proportion of missing values per feature combination"))
dev.off() 

# # Plot scatterplot matrix of missing
# pdf("scatter-matrix-missing.pdf", width=11.69, height=8.27)
# scattmatrixMiss(adult.train[c("workclass","occupation","native.country")],
#                 labels=c("Work class", "Occupation", "Native country"))
# dev.off() 

# # Barplot of missing
# pdf("barplot-occ-missing.pdf", width=11.69, height=8.27)
# barMiss(adult.train[c("workclass","occupation","native.country")],
#         pos=2,
#         selection = "any",
#         cex.lab=.6,
#         cex.axis=.5,
#         ylab = "Number of missing/observed in Occupation",
#         xlab = "")
# dev.off()

# Barplot of missing
adult.train$workclass.missing <- as.factor(ifelse(is.na(adult.train$workclass),1,0))
adult.train$occupation.missing <- as.factor(ifelse(is.na(adult.train$occupation),1,0))
adult.train$native.country.missing <- as.factor(ifelse(is.na(adult.train$native.country),1,0))

adult.train$occ.native.missing <- as.factor(ifelse(is.na(adult.train$occupation) | is.na(adult.train$native.country),1,0))
adult.train$work.native.missing <- as.factor(ifelse(is.na(adult.train$workclass) | is.na(adult.train$native.country),1,0))


workclass.missing <- ggplot(adult.train, aes(x = workclass, fill = occ.native.missing)) + 
  geom_bar(aes(y = (..count..)/sum(..count..))) + 
 # scale_y_continuous(limits=c(0, 0.75), breaks=c(0,0.25,0.5,0.75),labels=c("0", "25", "50", "75")) + 
  ylab("") + 
  xlab("") +
  ggtitle("Work class") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_discrete(name="Missing in
Occupation or Native country",
                      labels=c("No", "Yes"))

occupation.missing <- ggplot(adult.train, aes(x = occupation, fill = work.native.missing)) + 
  geom_bar(aes(y = (..count..)/sum(..count..))) + 
 # scale_y_continuous(limits=c(0, 0.25), breaks=c(0,0.25),labels=c("0", "25")) + 
  ylab("") + 
  xlab("") +
  ggtitle("Occupation") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_discrete(name="Missing in 
Work Class or Native country",
                      labels=c("No", "Yes"))

pdf("barplot-missing.pdf", width=11.69, height=8.27)
grid.arrange(workclass.missing, occupation.missing,
             ncol=1, nrow=2, left="Frequency (%)", bottom="")
dev.off()