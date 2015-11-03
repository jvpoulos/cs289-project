# Create plots for exploratory data analysis 

# Libraries
require(ggplot2)
require(GGally)

# Plot pairwise correlations and distributions for continuous features
png("pairs-plot-continuous.png", width=650, height=500)
ggpairs(adult.pre[["train.features"]][,c("age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week")], 
        upper="blank",
        columnLabels=c("Age","Sample wt.","Edu. no.", "Cap. gain", "Cap. loss", "Hrs./wk."))
dev.off() 

# Plot pairwise correlations and distributions for non-imputed binary variables that have missing values
png("pairs-plot-binary.png", width=650, height=500)
ggpairs(adult.pre[["train.features"]][,!colnames(adult.pre[["train.features"]]) %in% c("age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week")], 
        upper="blank",
        columnLabels=c("workclass","occupation","native.country"))
dev.off() # break up by cat variable compare missing vs. imputed

# Plot histogram for proportion of missing values per feature

adult.pre[["train.features"]]

