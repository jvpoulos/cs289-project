from os.path import splitext
import numpy as np
import matplotlib.pylab as plt
import seaborn as sbn
from mpl_toolkits.mplot3d import Axes3D

def savetxt_compact(fname, x, fmt="%.6g", delimiter=','):
    with open(fname, 'w') as fh:
        for row in x:
            line = delimiter.join("0" if value == 0 else fmt % value for value in row)
            fh.write(line + '\n')

# enumerate results files
result_files = (#'net_results_median.np',
                #'modern_net_results_median.np',
                # 'net_results_mean.np',
                #'modern_net_results_mean.np',
                #'net_drop_bin_scaled_results.np',
                'modern_net_drop_bin_scaled_results.np',
                #'net_replace_bin_scaled_results.np',
                #'modern_net_replace_bin_scaled_results.np',
                #'net_predicted_bin_scaled_results.np',
                'modern_net_predicted_bin_scaled_results.np',
                #'net_mode_bin_scaled_results.np',
                # 'modern_net_mode_bin_scaled_results.np',
                # 'net_facanal_bin_scaled_results.np',
                'modern_net_facanal_bin_scaled_results.np')

# plot and save results to png
for result_filename in result_files:
  # load results
  results = np.load(result_filename)
  savetxt_compact('./results/' + str(result_filename) + '_.csv', results, fmt='%.4f')

  # 2D plot
  g1_indices = np.where(results[:,1] == 0.0001)
  g2_indices = np.where(results[:,1] == 0.001)
  g3_indices = np.where(results[:,1] == 0.01)
  g4_indices = np.where(results[:,1] == 0.1)

  colors = ['b', 'c', 'g', 'r']

  g1 = plt.scatter(results[:,0][g1_indices], results[:,3][g1_indices],
                   marker='o', color=colors[0])
  g2 = plt.scatter(results[:,0][g2_indices], results[:,3][g2_indices],
                   marker='o', color=colors[1])
  g3 = plt.scatter(results[:,0][g3_indices], results[:,3][g3_indices],
                   marker='o', color=colors[2])
  g4 = plt.scatter(results[:,0][g4_indices], results[:,3][g4_indices],
                   marker='o', color=colors[3])

  plt.legend((g1,g2,g3,g4),
             (r'$\gamma=10^{-4}$',r'$\gamma=10^{-3}$',r'$\gamma=10^{-2}$', r'$\gamma=10^{-1}$'),
             scatterpoints=1,
             loc='lower left',
             ncol=4,
             fontsize=8)

  plt.xlabel('Alpha')
  plt.ylabel('Error rate')
  plt.savefig('./plots/{}_2d.png'.format(splitext(result_filename)))

  # 3D plot
  fig = plt.figure()
  ax = plt.subplot(111, projection='3d')

  ax.scatter(results[:,0][g1_indices], results[:,3][g1_indices],
             marker='o', color=colors[0], label=r'$\gamma=0.0001$')
  ax.scatter(results[:,0][g2_indices], results[:,3][g2_indices],
             marker='o', color=colors[1], label=r'$\gamma=0.001$')
  ax.scatter(results[:,0][g3_indices], results[:,3][g3_indices],
             marker='o', color=colors[2], label=r'$\gamma=0.01$')
  ax.scatter(results[:,0][g4_indices], results[:,3][g4_indices],
             marker='o', color=colors[3], label=r'$\gamma=0.1$')

  plt.legend(loc='upper left', numpoints=1, ncol=3,
             fontsize=8, bbox_to_anchor=(0, 0))

  ax.set_xlabel('Alpha')
  ax.set_zlabel('Gamma')
  ax.set_ylabel('Error rate')
  plt.savefig('./plots/{}_3d.png'.format(splitext(result_filename)))