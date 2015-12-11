from os.path import splitext
import numpy as np
import matplotlib.pylab as plt
import seaborn as sbn
from mpl_toolkits.mplot3d import Axes3D

# enumerate results files
result_files = np.loadtxt('result_filenames.np', dtype=str)

# plot and save results to png
for result_filename in result_files:
    # load results
    results = np.load(result_filename)

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
               (r'$\gamma=0.0001$',r'$\gamma=0.001$',r'$\gamma=0.01$', r'$\gamma=0.1$'),
               scatterpoints=1,
               loc='upper left',
               ncol=1,
               fontsize=8)

    plt.xlabel('Alpha')
    plt.ylabel('Error rate')
    plt.savefig('../plots/{}_2d.png'.format(splitext(result_filename)))

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
    plt.savefig('../images/{}_3d.png'.format(splitext(result_filename)))