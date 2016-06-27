import numpy.random as rnd
import numpy.linalg as la
import numpy as np

author = "y.selivonchyk"


class GausianClusters:
    """Class allows to draw inputs according to 2 n-dimensional gaussian clusters.
    Clusters intersect each other significantly"""

    def __init__(self, d=3, clusterscale=1, random_state=None, base=None):
        self.clusterscale = clusterscale
        self.d = d
        self.random_state = random_state

        if base is not None:
            if base.d < self.d:
                print "TTOOOOO BAAAAADDD"
                return

            self.s1 = base.s1[0:d]
            self.s2 = base.s2[0:d]
            self.shift = base.shift[0:d]
            self.shift *= np.sqrt(base.d)
            self.shift /= np.sqrt(d)
            print 'initialized with existing'
            return

        self.s1 = self.__get_scale()
        self.s2 = self.__get_scale()

        rand = np.abs(rnd.randn(self.d))            # random vector
        rand /= la.norm(rand)               # random unit vector
        self.shift = rand * self.s1         # random unit vector on surface s1
        rand = np.abs(rnd.randn(self.d))            # random vector
        rand /= la.norm(rand)               # random unit vector
        self.shift += rand * self.s2        # random unit vector on surface s1
        print 'shift before', self.shift, la.norm(self.shift)
        self.shift /= np.sqrt(self.d)       # get it closer to get some intersection in multydimensional environment
        # self.shift *= 3                     # so it would look better :)
        print 'Gausian clusters:'
        print 's1   :\t', self.s1, la.norm(self.s1)
        print 's2   :\t', self.s2, la.norm(self.s2)
        print 'shift:\t', self.shift, la.norm(self.shift)

    def __get_scale(self):
        s = rnd.lognormal(mean=0.0, sigma=1, size=self.d)
        s /= np.min(s)
        ratio = np.max(s) - 1
        s = ((s - 1) / ratio * self.clusterscale) + 1
        s /= la.norm(s)
        return s

    def __transform(self, x1, x2):
        return x1 * self.s1, x2 * self.s2 + self.shift

    def generate(self, n):
        x1, x2 = rnd.randn(n, self.d), rnd.randn(n, self.d)
        return self.__transform(x1, x2)

    def generate_classification(self, n, seed=-1):
        if seed != -1:
            np.random.seed(seed=seed)

        n /= 2
        x1, x2 = self.generate(n)
        return self.__to_classification(n, x1, x2)

    def __to_classification(self, n, x1, x2):
        x1_cls = np.ones((n, self.d+1))
        x1_cls[:, :-1] = x1
        x2_cls = np.zeros((n, self.d+1))
        x2_cls[:, :-1] = x2

        res = np.concatenate((x1_cls, x2_cls))
        rnd.shuffle(res)
        x, y = np.hsplit(res, [self.d])
        return x, y.transpose()[0]

    def generate_classification_outliers(self, n, sigma):
        total, res, c = 0, None, 0
        while total < n:
            c += 1
            z = rnd.randn(n, self.d) * 2
            nr = np.zeros((n, self.d + 1))
            nr[:, :-1] = z
            nr[:, -1] = la.norm(z, axis=1)
            # print la.norm(z, axis=1)
            nr = np.array(filter(lambda x: x[self.d] > sigma * np.sqrt(self.d), nr))
            if res is None and nr.shape[0] > 0:
                res = nr
            else:
                if nr.shape[0] > 0:
                    res = np.concatenate((res, nr))
                    total += nr.shape[0]
        print 'GC. Outlier generation took %d rounds' % c
        res, norm = np.hsplit(res, [self.d])
        x1, x2 = np.vsplit(res, [n/2])
        x2, x3 = np.vsplit(x2, [n/2])   # get rid of the tail
        x1, x2 = self.__transform(x1, x2)
        return self.__to_classification(n/2, x1, x2)

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
    def gamma(x):
        if x.shape[0] > 100:
            x = x[0:100]
        pt_sq_norms = (x ** 2).sum(axis=1)
        dists_sq = -2 * np.dot(x, x.T) + pt_sq_norms.reshape(-1, 1) + pt_sq_norms
        dists_sq[dists_sq < 0] = 0
        dist = np.sqrt(dists_sq)
        return 1/dist.mean()

    @staticmethod
    def pca(x):
        if x[0].size == 2:
            return x[:, 0], x[:, 1], 0
        return x[:, 0], x[:, 1], x[:, 2]