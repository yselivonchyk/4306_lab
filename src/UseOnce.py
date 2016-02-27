from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy.linalg as la
import numpy as np
import GausianClusters as gc
from matplotlib.mlab import PCA
import sklearn.metrics.pairwise.rbf_kernel as rbfk


gen = gc.GausianClusters(3, 3)
print gen.generate_classification(6)

n, d, s = 500, 3, 5
g = gc.GausianClusters(d, s)
x, y = g.generate_classification(1000)
print x.shape
example_size = 1000
if True:
    ax = plt.subplot(111)
    x = PCA(x)

    # 3D input plot
    # # show inputs
    # x1, x2 = generator.generate(500)
    # ax = plt.subplot(221, projection='3d')
    # marker = '.'
    # xp, yp, zp = gc.GausianClusters.pca(x1)
    # ax.scatter(xp, yp, zp, c='r', marker=marker)
    # xp, yp, zp = gc.GausianClusters.pca(x2)
    # ax.scatter(xp, yp, zp, c='b', marker=marker)
    # # print box to maintain the scales
    # xp, yp, zp = generator.box_3d()
    # ax.scatter(xp, yp, zp, '.', c='white', alpha=0.0)
    # if True:
    ds = x.Y, y
    x1, x2, y1, y2 = [], [], [], []
    for i in range(example_size):
        if ds[1][i] == 1:
            x1.append(ds[0][i][0])
            y1.append(ds[0][i][1])
        else:
            x2.append(ds[0][i][0])
            y2.append(ds[0][i][1])
    ax.plot(x1, y1, 'bs')
    ax.plot(x2, y2, 'r^')

plt.show()
