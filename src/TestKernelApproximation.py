import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import settings
import GausianClusters as gc
from sklearn.kernel_approximation import RBFSampler
import sklearn.metrics.pairwise as rbfk

import RKS_Sampler


def gram(x, gamma):
    sigma = 1. / np.sqrt(2*gamma)
    return Gaussian(x[None,:,:],x[:,None,:],sigma,axis=2)

def gram2(x, sigma):
    pt_sq_norms = (x ** 2).sum(axis=1)
    dists_sq = -2 * np.dot(x, x.T) + pt_sq_norms.reshape(-1, 1) + pt_sq_norms
    # turn into an RBF gram matrix
    gamma = 1.0/(sigma**2 * 2)
    km = np.sqrt(dists_sq) * (-1) * gamma
    print gamma, sigma
    return np.exp(km, km)  # exponentiates in-place

def Gaussian(x,z,sigma,axis=None):
    return np.exp((-(np.linalg.norm(x-z, axis=axis)**2))/(2*sigma**2))



def gram_projected(x):
    return np.dot(x, x.T)

x_label = "Projection size D (D = |w|)"
y_label = u"\u0394 = K(x,y) - z(x)z(y)"
y_label2 = u"E[\u0394] | \u0394 = abs(K(x,y) - z(x)z(y))"


def print_arrs(*args):
    i = 0
    for arr in args:
        print i, arr.shape
        print arr
        i += 1


# def compare(ref, *args):
#     for m in args:
#         dif = ref - m
#         # dif = dif / ref
#         dif = np.absolute(dif)
#         mean = dif.mean()
#         print "%.2f" % mean, mean


def mean_difference(ref, x):
    dif = ref - x
    dif = np.absolute(dif)
    return dif.mean()


def difference(ref, x):
    dif = ref - x
    dif = np.absolute(dif)
    return dif


def get_w_sizes(min_v=10, max_v=100000, intermediate=True):
    i = min_v
    s = []
    while i < max_v:
        s.append(i)
        if intermediate:
            s.append(int(i * np.sqrt(10)))
        i *= 10
    s.append(i)
    return s


# Plotting


def get_plot():
    plot = plt
    plot.autoscale(True)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.legend(loc='best')
    plot.grid(True)
    return plot


def savefig(name):
    plt.savefig(settings.plotLocation + name + '_.png', dpi=600)
    if settings.currentPlotLocation is not None:
        plt.savefig(settings.currentPlotLocation + name + '_.png', dpi=600)



# end Plotting


def test_once(x, wsizes, state, gamma=1):
    runs = []
    # wsizes = map(lambda y: y/2, wsizes)
    for wsize in wsizes:
        sampler = RKS_Sampler.RKSSampler(random_state=state, n_dim=wsize, gamma=gamma)
        features = sampler.transform_2n(x)
        # test for skikit implementation
        sampler = sk.kernel_approximation.RBFSampler(n_components=wsize, gamma=gamma)
        features = sampler.fit_transform(x)
        res = gram_projected(features)
        runs.append(res)
    return runs


def test_randomstate():
    gamma = 1
    wsizes = get_w_sizes(max_v=1000)
    plot = get_plot("")

    x = np.random.randn(40, 20)
    gm = gram(x, gamma)
    for i in range(10):
        runs = test_once(x, wsizes, i, gamma)
        runs = [mean_difference(gm, i) for i in runs]
        plot.semilogx(np.array(wsizes), np.array(runs), 'o')
    plot.xlim(0.9*wsizes[0], 1.1*wsizes[-1])

    savefig('tes')
    plot.title("Dependence on random state")
    plot.legend(range(10))


# Test: expected value of delta and confidence intervals

def expected_epsilon(delta, D):
    return np.sqrt(-2* np.log(delta/2)/D)


def expected_D(epsilon, delta):
    return np.ceil((-2) * np.log(delta/2) / (epsilon**2))


def print_predicted_vs_actual(wsizes, j, r, N, maxdelta=0.2, d=1):
    smallestDelta = 10.0 /(N**2/2)
    delta = maxdelta

    D = wsizes[j]
    eps_, expeps_, ratio_, delta_ = [],[],[], []
    print len(r)
    while delta >= smallestDelta:
        index = int(len(r) * (1.0-delta))
        epsilon = r[index]
        exp_epsilon = expected_epsilon(delta, D)
        exp_D = expected_D(epsilon, delta)
        print D, delta, exp_epsilon, epsilon, exp_D, index
        print 'D, delta, exp_eps, eps, exp_D, factor: %6d \t %.4f \t %.4f \t %.4f \t %6d \t\t %.4f \ti:%5d'\
              % (D, delta, exp_epsilon, epsilon, exp_D, exp_D*1.0/D, index)

        delta /= 2
        if delta < smallestDelta and delta != smallestDelta/2:
            delta = smallestDelta

        eps_.append(epsilon)
        expeps_.append(exp_epsilon)
        ratio_.append(D*1.0/exp_D)
        delta_.append(delta)
    print eps_
    print expeps_

    plt.semilogx(np.array(delta_), np.array(eps_), '-', label="actual error(delta)")
    plt.semilogx(np.array(delta_), np.array(expeps_), '-', label="expected error(delta)")
    plt.semilogx(np.array(delta_), np.array(ratio_), '-', label="fraction of D/D_expected")
    plt.legend(fancybox=True, framealpha=0.5, loc='upper left')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.xlabel('Delta [%.2f, %.8f]' %(maxdelta, smallestDelta))
    savefig("inequalities%s_%d" % (str(N), d))
    plt.title("Inequalities")


def test_inequalities(d=50, N=2000, maxdelta=0.8, gamma=1, wsizes=np.array([100])):
    print "\n\r\n\r\n\r>>>TEST INEQUALITIES<<<"
    gamma = gamma
    x = np.random.rand(N, d)
    # print x, 'x'
    gm = gram(x, gamma)
    runs = test_once(x, wsizes, 1, gamma)
    runs_difference = [prepare_for_inteval(x, gm) for x in runs]

    if len(wsizes) == 1:
        print_predicted_vs_actual(wsizes, 0, runs_difference[0], N)
    else:
        for i, run in runs_difference:
            print_predicted_vs_actual(wsizes, j, run, N, maxdelta, d)
    print "\n\r\n\r\n\r>>>TEST INEQUALITIES DONE<<<"

            # plt.ylim(-0.2, 0.2)


def test_precision_interval(d=100, N=1000, gamma=1):
    print "\n\r\n\r\n\r>>>TEST PRECISION INTERVALS<<<\n\r"
    gamma = gamma
    x = np.random.rand(N, d)
    # print x, 'x'
    wsizes = get_w_sizes(min_v=100, intermediate=False)
    gm = gram(x, gamma)
    runs = test_once(x, wsizes, 1, gamma)
    runs_difference = [prepare_for_inteval(x, gm) for x in runs]

    wsizes = [np.log(x)/np.log(10) for x in wsizes]
    labels = ['D=10^' + str(np.round((i+0.05)*10)/10.0) for i in wsizes]
    print labels
    plot_intervals(x_label, y_label, runs_difference, labels, True)

    ax = plt.axes()
    ax.yaxis.grid()
    savefig("test_precision_interval_error1")
    plt.title("Absolute estimation error (50% and 90% intervals)")

    plt.figure()
    print labels
    plot_intervals(x_label, y_label, runs_difference, labels, False)        #
    plt.legend(fancybox=True, framealpha=0.5, loc='upper left')
    ax = plt.axes()
    ax.yaxis.grid()
    plt.figtext(0.55, 0.85, 'Absolute error intervals', fontsize=14, backgroundcolor='#DDDDDD')
    savefig("test_precision_interval_error2")
    plt.title("Absolute estimation error (50% and 90% intervals)")
    plt.figure()

    kernel_values = [prepare_for_intervals_no_difference(r) for r in runs]
    kernel_values.insert(0, prepare_for_intervals_no_difference(gm))

    labels.insert(0, "K(x,y)")
    plot_intervals(x_label, "K(x, y) and z'(x)z(y) values", kernel_values, labels, True, [5, 95])   #
    plt.figtext(0.55, 0.85, 'Intervals of kernel values', fontsize=14, backgroundcolor='#DDDDDD')
    plt.ylim(-0.3, 0.3)
    ax = plt.axes()
    ax.yaxis.grid()
    savefig("test_precision_interval_delta")
    plt.title("Actual and estimated values for single input set")

    # plt.ylim(-0.2, 0.2)


def prepare_for_inteval(x, gm):
    x = difference(gm, x)
    x = np.reshape(x, newshape=-1)
    x = np.sort(x)
    x = x[0::2]   # ignore duplicates
    return x


def prepare_for_intervals_no_difference(x):
    x = np.reshape(x, newshape=-1)
    x = np.sort(x)
    x = x[0::2]
    return x


def plot_intervals(lbl_x, lbl_y, data, labels, printmax, whis=[0, 90], label=''):
    outlierMarker = 'x' if printmax else ''
    plt.boxplot(data, 0, outlierMarker, whis=whis, labels=labels)
    # plt.boxplot(data, 0, outlierMarker, whis=whis)
    plt.xlabel(lbl_x)
    plt.ylabel(lbl_y)


# input size test


def test_input_size():
    gamma = 1
    wsizes = get_w_sizes(max_v=1000)
    plot = get_plot()

    for in_size in range(10, 50, 10):
        np.random.seed(1)
        x = np.random.randn(10, in_size)
        gm = gram(x, gamma)
        runs = test_once(x, wsizes, 1, gamma)
        runs = [mean_difference(gm, i) for i in runs]
        plot.semilogx(np.array(wsizes), np.array(runs), '-', label=in_size)
    plot.xlim(0.9*wsizes[0], 1.1*wsizes[-1])
    plt.title("Dependence on input size")
    plt.legend()


# denormolized


def draw_denormolized_experiment(x, wsizes, label):
    gamma = 1
    gm = gram(x, gamma)
    runs = test_once(x, wsizes, None, gamma)
    runs = [mean_difference(gm, i) for i in runs]
    plt.semilogx(np.array(wsizes), np.array(runs), '-', label=label)


def test_denormolized(max_w=100000):
    print "\n\r\n\r\n\r>>>TEST DENORMALIZATION<<<"
    wsizes = get_w_sizes(max_v=max_w)
    plot = get_plot()
    plot.ylabel(y_label2)

    input_size = 100
    vector_size = 5

    np.random.seed(1)
    x_original = np.random.randn(input_size, vector_size)
    draw_denormolized_experiment(x_original, wsizes, "Normalized")

    scale = np.random.rand(vector_size) * 100
    x_scaled = np.multiply(scale, x_original)
    draw_denormolized_experiment(x_scaled, wsizes, "Scaled randomly")

    shift = (np.random.rand(vector_size) - 0.5) * 1000
    x = x_original + shift
    draw_denormolized_experiment(x, wsizes, "Shifted")

    x = np.random.rand(input_size, vector_size)
    draw_denormolized_experiment(x, wsizes, u"uniform x \u2208[0,1]")

    # x = np.random.rand(input_size, vector_size) * 2
    # draw_denormolized_experiment(x, wsizes, u"uniform x \u2208[0,2]")
    #
    # x = np.random.rand(input_size, vector_size) * 4
    # draw_denormolized_experiment(x, wsizes, u"uniform x \u2208[0,4]")

    # plot.xlim(0.9*wsizes[0], 1.1*wsizes[-1])
    # plot.ylim(0.0, 0.35)
    plt.legend()
    savefig("fig_input")
    plt.title("Dependence on input data")



# test max
def prob_of_maxdelta(m, delta, D, d, gamma):
    m1 = np.power(2, 8)
    m2 = np.square(gamma * m / delta)
    m3 = np.exp(-1 * D * delta * delta / (4*(d+2)))
    return m1*m2*m3


def test_max():
    gamma = 1
    wsizes = get_w_sizes(min_v=1000)
    plot = get_plot()

    x = np.random.randn(100, 100)
    gm = gram(x, gamma)
    runs = test_once(x, wsizes, 10, gamma)
    means = [mean_difference(gm, i) for i in runs]
    plot.semilogx(np.array(wsizes), means, '-', label='mean')
    max = [np.amax(difference(gm, a)) for a in runs]
    plot.semilogx(np.array(wsizes), max, 'o', label='max')

    # calculate probability of greater delta
    gm[gm == 0] = 1                     # replace zeros
    m = np.amax(np.amin(gm, axis=1))    # e-net d
    print "M", m
    probabilities = []
    for i in range(len(max)):
        p = prob_of_maxdelta(1, max[i], wsizes[i], len(x), 1)
        probabilities.append(p)
    print probabilities
    plot.xlim(0.9*wsizes[0], 1.1*wsizes[-1])
    # plot.legend()


def GaussianMatrix(X,gamma):
    row,col=X.shape
    GassMatrix=np.zeros(shape=(row,row))
    X=np.asarray(X)
    i=0
    for v_i in X:
        j=0
        for v_j in X:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,gamma)
            j+=1
        i+=1
    return GassMatrix


def run_all_relevant(faste=False):
    if faste:
        test_precision_interval(N=100, gamma=0.2)
        plt.figure()
        # test_inequalities(N=200)
        # plt.figure()
        # test_denormolized(max_w=1000)
    else:
        test_precision_interval(gamma=0.2)
        plt.figure()
        test_inequalities()
        plt.figure()
        test_denormolized()
    print '\n\r\n\r\n\rRELEVANT TEST FINISHED\n\r\n\r'

# x = np.random.rand(100, 100)
# gamma = 0.001
# g = []
# for i in range(20):
#     gamma *=2
#     gx = gram(x, gamma =gamma)
#     print gram(x, gamma =gamma)
#     s = RKS_Sampler.RKSSampler(n_dim=100, gamma=gamma)
#
#     gm =  gram_projected(s.transform_cos(x))
#     print gram_projected(s.transform_cos(x))
#
#     s = RBFSampler(gamma=gamma)
#     gt = gram_projected(s.fit_transform(x))
#     print gram_projected(s.fit_transform(x))
#
#     gm = np.absolute(gm - gx)
#     gt = np.absolute(gt - gx)
#     g.append(gm.mean()/gt.mean())
#     print '%.4f \t %.4f' % (gamma, (gm.mean()/gt.mean()))
# print g
# print np.array(g).mean()

# test_randomstate()
# plt.figure()
# test_precision_interval(d=100, N=1000, gamma=0.2)
# test_inequalities(d=50, N=2000, gamma=1, wsizes=np.array([100]), maxdelta=0.8)
# test_input_size()
# plt.figure()
# test_denormolized()
# # plt.figure()
# test_max()

#
# test_inequalities(d=6)
# plt.show()



