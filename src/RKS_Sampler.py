from itertools import *
from numpy.random import *
from numpy import dot
from numpy import cos, sin
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot

author = "y.selivonchyk"


class RKSSampler:
    """Random Kitchen Sinks sampler implementation"""

    def __init__(self, random_state=None, n_dim=100, gamma=1.0):
        self.w = None
        self.b = None
        self.gamma = gamma
        self.input_dim = 0
        self.n_dim = n_dim
        self.random_state = random_state

    def transform_2n(self, X):
        self.__generate_w(X)
        x_cos = cos(dot(X, self.w))
        x_sin = sin(dot(X, self.w))
        res = np.concatenate((x_cos, x_sin), axis=1)
        res /= np.sqrt(self.n_dim)
        return res

    def transform_rks(self, X):
        self.__generate_w(X)
        return np.exp(1j * dot(X, self.w)) / np.sqrt(self.n_dim)

    # cos(wx + b)
    def transform_cos(self, X):
        self.__generate_w(X)
        # projection = dot(X, self.w) + self.b
        # return cos(projection) * np.sqrt(2) / np.sqrt(self.n_dim)

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.w)
        projection += self.b
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_dim)
        return projection

    def __generate_w(self, X):
        if not self.w is None:
            return

        self.input_dim = len(X[0]) if len(X.shape) == 2 else len(X)
        if not(self.random_state is None):
            seed(self.random_state)
        self.w = randn(self.input_dim, self.n_dim) * np.sqrt(2 * self.gamma)
        self.b = random(self.n_dim) * 2 * np.pi     # b belongs to [0, 2pi)

    @staticmethod
    def mult(w, x):
        return sum(imap(operator.mul, w, x))


