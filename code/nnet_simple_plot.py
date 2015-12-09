import os.path
import numpy as np
import matplotlib.pylab as plt
import seaborn as sbn
from mpl_toolkits.mplot3d import Axes3D

if not os.path.isfile('net_results.np'):
    import net

results = np.load('net_results.np')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(results[:,0],results[:,3],
           c=results[:,1], cmap='Spectral',
           label=np.unique(results[:,1]))
plt.legend()
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