# Create plots for exploratory data analysis 

# Libraries
require(ggplot2)
require(GGally)

patient <- FALSE

if(patient){
# Plot pairwise correlations and distributions for continuous features
png("pairs-plot-continuous.png", width=650, height=500)
ggpairs(adult.pre[["train.features"]][,c("age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week")], 
        upper="blank",
        columnLabels=c("Age","Sample wt.","Edu. no.", "Cap. gain", "Cap. loss", "Hrs./wk."))
dev.off() 

# # Plot pairwise correlations and distributions for workclass & occupation
# png("pairs-plot-binary.png", width=650, height=500)
# ggpairs(adult.pre[["train.features"]][,c("workclass.Federal.gov", "workclass.Local.gov", "workclass.Never.worked", "workclass.Self.emp.inc", "workclass.Self.emp.not.inc", "workclass.State.gov", "workclass.Without.pay",
#                                          "occupation.Adm.clerical", "occupation.Armed.Forces", "occupation.Craft.repair", "occupation.Exec.managerial", "occupation.Farming.fishing", "occupation.Handlers.cleaners",
#                                          "occupation.Machine.op.inspct", "occupation.Other.service", "occupation.Priv.house.serv", "occupation.Prof.specialty", "occupation.Protective.serv", "occupation.Sales", "occupation.Tech.support", "occupation.Transport.moving")], 
#         upper="blank",
#         columnLabels=c("Federal-gov", "Local-gov", "Never-worked"," Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay",
#                        "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners",
#                        "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"))
# dev.off() 
}

# Plot histogram for proportion of missing values per feature

adult.pre[["train.features"]]

