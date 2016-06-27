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
n = 40
d = 3.5

rnd.seed(20)
x1 = rnd.randn(N, 2)
x1 =np.add(x1, np.array([0, d]))
x2 = rnd.randn(N, 2)
x2 = np.add(x2, np.array([d*1.5, 0]))
xt = np.concatenate((x1[n:-1,], x2[n:-1,]))
yt = np.concatenate((np.ones(N-n-1), -1*np.ones(N-n-1)))
X = np.concatenate((x1[0:n,], x2[0:n,]))
y = np.concatenate((np.ones(n), -1*np.ones(n)))


iris = datasets.load_iris()
h = .02  # step size in the mesh
C = 2.0  # SVM regularization parameter
# svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=2.9, C=C).fit(X, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.SVC(kernel='linear', C=C).fit(X, y)

print lin_svc.coef_
print lin_svc.get_params()

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


for i, clf in enumerate((lin_svc, lin_svc)):
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    # print X, y
    # plt.scatter(xt[:, 0], xt[:, 1], c=yt, cmap=plt.cm.Paired, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    # print xt.shape, yt.shape
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Data')
    savefig('vis_lin_data')
    break

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


for i, clf in enumerate((lin_svc, lin_svc)):
    w = lin_svc.coef_[0]
    print(w)
    a = -w[0] / w[1]
    xl = np.linspace(x_min, x_max, 200)
    yl = a * xl - lin_svc.intercept_[0] / w[1]

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=1, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot also the training points
    # print X, y
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    # print xt.shape, yt.shape
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    de = 3.1*0.7
    plt.plot(xl, yl+de, 'k--', label='decision margin')
    plt.plot(xl, yl, 'k-', label='decision boundary')
    plt.plot(xl, yl-de, 'k--')
    plt.legend()
    plt.title('Linear SVM. Decision boundary.')
    savefig('vis_lin_margin')
    break

for i, clf in enumerate((lin_svc, lin_svc)):
    w = lin_svc.coef_[0]
    print(w)
    a = -w[0] / w[1]
    xl = np.linspace(x_min, x_max, 200)
    yl = a * xl - lin_svc.intercept_[0] / w[1]

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
    # print X, y
    plt.scatter(xt[:, 0], xt[:, 1], c=yt, cmap=plt.cm.Paired, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    # print xt.shape, yt.shape
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('Linear SVM. Prediction.')
    savefig('vis_lin_prediction')
    break

# plt.show()