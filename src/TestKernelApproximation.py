import numpy as np
import RKS_Sampler
import matplotlib.pyplot as plt
import sklearn.kernel_approximation as sk


def gram(x, sigma):
    pt_sq_norms = (x**2).sum(axis=1)
    dists_sq = -2 * np.dot(x, x.T) + pt_sq_norms.reshape(-1, 1) + pt_sq_norms
    km = dists_sq * (-sigma / 2)
    return np.exp(km, km)  # exponentiates in-place


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


def get_plot(title):
    plot = plt
    plot.autoscale(True)
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.legend(loc='best')
    plot.grid(True)
    return plot


def savefig(name):
    plt.savefig('./plots/' + name + '_.png')


# end Plotting


def test_once(x, wsizes, state, sigma=1):
    runs = []
    # wsizes = map(lambda y: y/2, wsizes)
    for wsize in wsizes:
        sampler = RKS_Sampler.RKSSampler(random_state=state, n_dim=wsize, sigma=sigma)
        features = sampler.transform_2n(x)
        # test for skikit implementation
        # sampler = sk.RBFSampler(n_components=wsize, gamma=sigma)
        # features = sampler.fit_transform(x)
        res = gram_projected(features)
        runs.append(res)
    return runs


def test_randomstate():
    sigma = 1
    wsizes = get_w_sizes(max_v=1000)
    plot = get_plot("Dependence on random state")

    x = np.random.randn(40, 20)
    gm = gram(x, sigma)
    for i in range(10):
        runs = test_once(x, wsizes, i, sigma)
        runs = [mean_difference(gm, i) for i in runs]
        plot.semilogx(np.array(wsizes), np.array(runs), 'o')
    plot.xlim(0.9*wsizes[0], 1.1*wsizes[-1])
    plot.legend(range(10))


# Test: expected value of delta and confidence intervals


def test_precision_interval():
    sigma = 1
    x = np.random.randn(100, 10) * 2
    # print x, 'x'
    wsizes = get_w_sizes(min_v=100, intermediate=False)
    gm = gram(x, sigma)
    runs = test_once(x, wsizes, 1, sigma)
    runs_difference = [prepare_for_inteval(x, gm) for x in runs]
    wsizes = [np.log(x)/np.log(10) for x in wsizes]
    labels = ['10^' + str(np.round((i+0.05)*10)/10.0) for i in wsizes]

    plt.title("Absolute estimation error (50% and 90% intervals)")
    plot_intervals(x_label, y_label, runs_difference, labels, True)
    savefig("fig_error_1")
    plt.figure()
    print labels
    plt.title("Absolute estimation error (50% and 90% intervals)")
    plot_intervals(x_label, y_label, runs_difference, labels, False)
    savefig("fig_error_2")
    plt.figure()

    plt.title("Actual and estimated values for single input set")
    kernel_values = [prepare_for_intervals_no_difference(r) for r in runs]
    kernel_values.insert(0, prepare_for_intervals_no_difference(gm))
    labels.insert(0, "K(x,y)")

    plot_intervals(x_label, "K(x, y) and z'(x)z(y) values", kernel_values, labels, True, [5, 95])
    plt.ylim(-0.3, 0.3)
    savefig("fig_data")
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


def plot_intervals(lbl_x, lbl_y, data, labels, printmax, whis=[0, 90]):
    outlierMarker = 'x' if printmax else ''
    plt.boxplot(data, 0, outlierMarker, whis=whis, labels=labels)
    plt.xlabel(lbl_x)
    plt.ylabel(lbl_y)


# input size test


def test_input_size():
    sigma = 1
    wsizes = get_w_sizes(max_v=1000)
    plot = get_plot("Dependence on input size")

    for in_size in range(10, 50, 10):
        np.random.seed(1)
        x = np.random.randn(10, in_size)
        gm = gram(x, sigma)
        runs = test_once(x, wsizes, 1, sigma)
        runs = [mean_difference(gm, i) for i in runs]
        plot.semilogx(np.array(wsizes), np.array(runs), '-', label=in_size)
    plot.xlim(0.9*wsizes[0], 1.1*wsizes[-1])
    plt.legend()


# denormolized


def draw_denormolized_experiment(x, wsizes, label):
    sigma = 1
    gm = gram(x, sigma)
    runs = test_once(x, wsizes, None, sigma)
    runs = [mean_difference(gm, i) for i in runs]
    plt.semilogx(np.array(wsizes), np.array(runs), '-', label=label)


def test_denormolized():
    wsizes = get_w_sizes(max_v=100000)
    plot = get_plot("Dependence on input data")
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


# test max
def prob_of_maxdelta(m, delta, D, d, sigma):
    m1 = np.power(2, 8)
    m2 = np.square(sigma * m / delta)
    m3 = np.exp(-1 * D * delta * delta / (4*(d+2)))
    return m1*m2*m3


def test_max():
    sigma = 1
    wsizes = get_w_sizes(min_v=1000)
    plot = get_plot("Dependence on random state")

    x = np.random.randn(100, 100)
    gm = gram(x, sigma)
    runs = test_once(x, wsizes, 10, sigma)
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
    plot.legend()


# test_randomstate()
# plt.figure()
test_precision_interval()
plt.figure()
# test_input_size()
# plt.figure()
test_denormolized()
# plt.figure()
# test_max()
# plt.show()



# np.random.seed(1)
# X = np.random.randn(100, 100)
# X = np.array([[1, 2], [3, 4]])
#
# n = 1000
# sampler = RKSSampler2.RKSSampler2(random_state=1, n_dim=n)
# ref_sampler = ka.RBFSampler(gamma=1, n_components=n, random_state=1)
#
# gram_m = gram(X, 1.0)
# proj_2n = sampler.transform_2n(X)
# proj_cos = sampler.transform_cos(X)
# proj_ = sampler.transform_rks(X)
# ref_proj = ref_sampler.fit_transform(X)
#
# proj_2n = np.dot(proj_2n, proj_2n.T)
# proj_cos = np.dot(proj_cos, proj_cos.T)
# proj_ = np.dot(proj_, proj_.T)
# ref_proj = np.dot(ref_proj, ref_proj.T)
#
# print_arrs(proj_2n, proj_cos, ref_proj, gram_m)
# compare(gram_m, proj_2n, proj_cos, ref_proj)


