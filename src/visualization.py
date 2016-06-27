print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import settings
from sklearn import svm, datasets


def savefig(name, postfix='', fig=None):
    if fig is None:
        fig = plt

    fig.savefig(settings.plotLocation + name + '_.png', dpi=300)
    if settings.currentPlotLocation is not None:
        fig.savefig(settings.currentPlotLocation + name + '_.png', dpi=300)


settings.init(True)


N = 3000
n = 1000
d = 5
x = rnd.randn(N*10, 2)*d*np.array([1, 1])
x = x[(4 > x[:,1]) & (x[:,1] > -4) & (4 > x[:,0]) & (x[:,0] > -4)]
x = x[:N,]
print x.shape
x = np.add(x, np.array([0, 0]))
y = np.ones(N)
# for i in range(N):
#     if x[i,0]**2 + x[i,1]**2 > np.abs(x[i,0]**3):
#         y[i] = -1
#
# x2 = np.ones((N, 2), dtype=np.float32)
# for i in range(N):
#     x2[i, 0], x2[i, 1] = x[i,0]**2 + x[i,1]**2, np.abs(x[i,0]**3)
# y2 = y

for i in range(N):
    if abs(x[i,0]) + abs(x[i,1])-0.25 > np.abs(x[i,0]**2):
        y[i] = -1

x2 = np.ones((N, 2), dtype=np.float32)
for i in range(N):
    x2[i, 0], x2[i, 1] = abs(x[i,0]) + abs(x[i,1])-0.25, np.abs(x[i,0]**2)
y2 = y

#
# x = np.add(x, np.array([0.66, 0.33]))

fig = plt.figure()
DefaultSize = fig.get_size_inches()
fig.set_size_inches((DefaultSize[0] * 2, DefaultSize[1] * 1))
# x1 = rnd.randn(N, 2)
# x1 =np.add(x1, np.array([0, d]))
# x2 = rnd.randn(N, 2)
# x2 = np.add(x2, np.array([d*1.5, 0]))
# xt = np.concatenate((x1[n:-1,], x2[n:-1,]))
# yt = np.concatenate((np.ones(N-n-1), -1*np.ones(N-n-1)))
# X = np.concatenate((x1[0:n,], x2[0:n,]))
# y = np.concatenate((np.ones(n), -1*np.ones(n)))

xt = x[n:-1,]
yt = y[n:-1,]
X = x[0:n,]
y = y[0:n,]
print X
print y

# import some data to play with
iris = datasets.load_iris()
# X = iris.data[:, :2]  # we only take the first two features. We could
#                       # avoid this ugly slicing by using a two-dim dataset
# y = iris.target

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=10, C=C).fit(X, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(x2[:n,], y2[:n])

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

xl = np.linspace(0, x_max, 100)
yl = xl**2-xl+0.25
for i, clf in enumerate((rbf_svc, lin_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Data')
    plt.plot(xl, yl, 'k-', label='decision boundary')
    plt.plot(xl, -yl, 'k-')
    plt.plot(-xl, yl, 'k-')
    plt.plot(-xl, -yl, 'k-')
    plt.legend()
    savefig('vis_nonlin_data')
    break

print 'nonlin'
for i, clf in enumerate((rbf_svc, lin_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.7)

    # Plot also the training points
    plt.scatter(xt[:, 0], xt[:, 1], c=yt, cmap=plt.cm.Paired, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Nonlinear SVM using RBF kernel')
    savefig('vis_nonlin_nonlin')
    break

x_min, x_max = x2[:, 0].min() - 1, x2[:, 0].max() + 1
y_min, y_max = x2[:, 1].min() - 1, x2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
print 'lin'
for i, clf in enumerate((lin_svc, lin_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=1, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.7)

    # Plot also the training points
    plt.scatter(x2[n:-1, 0], x2[n:-1, 1], c=y2[n:-1], cmap=plt.cm.Paired, alpha=0.25)
    plt.scatter(x2[:n, 0], x2[:n, 1], c=y2[:n], cmap=plt.cm.Paired)
    plt.xlabel('z1 = abs(x1)+abs(x2)')
    plt.ylabel('z2 = x1^2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Linear SVM using features z1, z2')
    savefig('vis_nonlin_lin')
    break

# plt.show()