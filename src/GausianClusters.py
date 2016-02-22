import numpy.random as rnd
import numpy.linalg as la
import numpy as np

author = "y.selivonchyk"


class GausianClusters:
    """Class allows to draw inputs according to 2 n-dimensional gaussian clusters.
    Clusters intersect each other significantly"""

    def __init__(self, d=3, clusterscale=1, random_state=None):
        self.clusterscale = clusterscale
        self.d = d
        self.random_state = random_state

        self.s1 = self.__get_scale()
        self.s2 = self.__get_scale()

        self.shift = rnd.randn(self.d)    # random vector
        self.shift /= la.norm(self.shift) # random unit vector
        self.shift *= self.s1             # random unit vector on random surface
        self.shift /= np.sqrt(self.d)     # get it closer to get some intersection in multydimensional environment
        self.shift *= 3                   # so it would look better :)
        print 'Gausian. secons cluster shift:', self.shift

    def __get_scale(self):
        s = rnd.lognormal(mean=0.0, sigma=1, size=self.d)
        s /= np.min(s)
        ratio = np.max(s)
        s = ((s - 1) / ratio * self.clusterscale) + 1
        s /= la.norm(s)
        return s

    def __transform(self, x1, x2):
        return x1 * self.s1, x2 * self.s2 + self.shift

    def generate(self, n):
        x1, x2 = rnd.randn(n, self.d), rnd.randn(n, self.d)
        return self.__transform(x1, x2)

    def generate_classification(self, n):
        n /= 2
        x1, x2 = self.generate(n)

        x1_cls = np.ones((n, self.d+1))
        x1_cls[:, :-1] = x1
        x2_cls = np.zeros((n, self.d+1))
        x2_cls[:, :-1] = x2

        res = np.concatenate((x1_cls, x2_cls))
        rnd.shuffle(res)
        x, y = np.hsplit(res, [self.d])
        return x, y.transpose()[0]

    def generate_unit(self, n):
        x1, x2 = rnd.randn(n, self.d), rnd.randn(n, self.d)
        x1 /= la.norm(x1, axis=1)[:,None]
        x2 /= la.norm(x2, axis=1)[:,None]
        return self.__transform(x1, x2)

    def box_3d(self):
        """
        :return: coordinates of a box to be drawn to maintain the aspect ration of 3d figure
        """
        box = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype='f')
        box_scale = np.max(self.s1)
        arr = np.array([box_scale, np.max(self.s2 + np.abs(self.shift))])
        box_scale = np.max(arr)
        return self.pca(box*box_scale)

    @staticmethod
    def pca(x):
        if x[0].size == 2:
            return x[:, 0], x[:, 1], 0
        return x[:, 0], x[:, 1], x[:, 2]