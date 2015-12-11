from os.path import splitext
import numpy as np
import matplotlib.pylab as plt
import seaborn as sbn

# enumerate results files
result_files = np.loadtxt('result_filenames.np', dtype=str)

model_tech_time = {}

# plot and save results to png
for result_filename in result_files:
    # save ave k-fold cpu time to dictionary
    model_tech_time[splitext(result_filename)] = np.load(result_filename)[-1]

plt.bar(range(len(model_tech_time)),
        model_tech_time.values(),
        align='center')

plt.xticks(range(len(model_tech_time)),
           model_tech_time.keys())

plt.savefig('../images/model_tech_time.png')