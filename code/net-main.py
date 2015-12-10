# Run net.py for each imputation method

imputation_methods = ["median", "mean"]

for imp_method in imputation_methods:
    execfile("load_data.py")
    execfile("net.py")