import os.path
import numpy as np
import matplotlib.pylab as plt
import seaborn as sbn
from mpl_toolkits.mplot3d import Axes3D

if not os.path.isfile('net_results.np'):
    import net

results = np.load('net_results.np')

"""
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(results[:,0],results[:,3],
           c=results[:,1], cmap='Spectral',
           label=np.unique(results[:,1]))

ax.set_xlabel('Alpha')
ax.set_ylabel('Error rate')

plt.legend()
plt.show()
"""
# 2D plot
g1_indices = np.where(results[:,1] == 0.0001)
g2_indices = np.where(results[:,1] == 0.001)
g3_indices = np.where(results[:,1] == 0.01)
g4_indices = np.where(results[:,1] == 0.1)

colors = ['b', 'c', 'g', 'r']

g1 = plt.scatter(results[:,0][g1_indices], results[:,3][g1_indices], marker='o', color=colors[0])
g2 = plt.scatter(results[:,0][g2_indices], results[:,3][g2_indices], marker='o', color=colors[1])
g3 = plt.scatter(results[:,0][g3_indices], results[:,3][g3_indices], marker='o', color=colors[2])
g4 = plt.scatter(results[:,0][g4_indices], results[:,3][g4_indices], marker='o', color=colors[3])

plt.legend((g1,g2,g3,g4),
           (r'$\gamma=0.0001$',r'$\gamma=0.001$',r'$\gamma=0.01$', r'$\gamma=0.1$'),
           scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=8)

plt.xlabel('Alpha')
plt.ylabel('Error rate')

plt.show()

# 3D plot

ax = plt.subplot(111, projection='3d')

ax.scatter(results[:,0][g1_indices], results[:,3][g1_indices], marker='o', color=colors[0],
        label=r'$\gamma=0.0001$')
ax.scatter(results[:,0][g2_indices], results[:,3][g2_indices], marker='o', color=colors[1],
        label=r'$\gamma=0.001$')
ax.scatter(results[:,0][g3_indices], results[:,3][g3_indices], marker='o', color=colors[2],
        label=r'$\gamma=0.01$')
ax.scatter(results[:,0][g4_indices], results[:,3][g4_indices], marker='o', color=colors[3],
        label=r'$\gamma=0.1$')

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

ax.set_xlabel('Alpha')
ax.set_zlabel('Gamma')
ax.set_ylabel('Error rate')

plt.show()

"""
# 3d-plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u_g, i_g = np.unique(results[:,1], return_inverse=True)

ax.scatter(results[:,0], i_g, results[:,3],
           zdir='z', s=20, c='b', marker='o', depthshade=True)

ax.set_xlabel('Alpha')
ax.set_ylabel('Gamma')
ax.set_zlabel('Error rate')

ax.set_yticks(np.arange(len(np.unique(results[:,1]))))
ax.set_yticklabels(u_g)

plt.show()
"""