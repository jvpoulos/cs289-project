# Run modern_net.py for each imputation method

imputation_methods = ["median", "mean"]

for imp_method in imputation_methods:
    execfile("load_data.py")
    execfile("modern_net.py")