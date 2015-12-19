import numpy as np
import time
from sklearn.svm import SVC
from sklearn import datasets
from numpy import random
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Ridge
import RKS_Sampler


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print np.concatenate((a, b), axis=1)

exit(0)

cov = datasets.fetch_covtype()
print cov.data.shape


def measure(func, arg, legend="time:"):
    start = time.clock()
    out = func(arg)
    end = time.clock()
    print legend, "\t", end - start
    return out

n = 10000
size = 5000
X = cov.data[0:n]
y = cov.target[0:n]
X_c = cov.data[0:n*2]
y_c = cov.target[0:n*2]
print X, y, "--"

rbf_sampler = RBFSampler(gamma=1, n_components=size, random_state=1)
sampler_rks = RKS_Sampler.RKSSampler(1, n_dim=size)
# measure(rbf_feature.fit_transform, X, "default")
X_features = rbf_sampler.fit_transform(X)
print "their", X_features[0]

# measure(rks_f.transform, X, "mine   ")
rks_features = sampler_rks.transform(X)
print "mine", rks_features[0]




clf = SGDClassifier()
clf.fit(X_features, y)
print "RBF", clf.score(rbf_sampler.fit_transform(X_c), y_c)

clf = Ridge(alpha=1.0)
clf.fit(rks_features, y)
print "RKS", clf.score(sampler_rks.transform(X_c), y_c)


