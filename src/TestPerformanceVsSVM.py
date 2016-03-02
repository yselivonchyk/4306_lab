import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import sklearn.kernel_approximation as smp
import sys
from sklearn import datasets as ds
from sklearn import svm
from sklearn import cross_validation
from RKS_Sampler import RKSSampler
from matplotlib import pyplot as plt
import random as rnd
import inspect
import GausianClusters as gc
print inspect.getmodule(np.dot)
import numpy.linalg as la
from matplotlib.mlab import PCA
from sklearn.datasets import make_classification



def get_w_sizes(min_v=10, max_v=10000, intermediate=0, linear=True):
    i = min_v
    s = []
    while i < max_v:
        s.append(i)
        for j in range(intermediate):
            if (i * np.power(10, (j+1)*1.0/(intermediate+1))) < max_v:
                # print i * np.power(10, (j+1)*1.0/(intermediate+1)), max_v
                val = int(i * np.power(10, (j+1)*1.0/(intermediate+1)))
                if val < max_v:
                    s.append(val)
                else:
                    print 'illigal wsize: ', val
        i *= 10
    if linear:
        last = s[-1]
        s[-1] = (last + s[-2])
        s.append(last)
    print 'w sizes: ', s
    return s


def get_log_sequence(base, start, stop, intermediate=0):
    s = []
    for i in range(start, stop):
        s.append(np.power(base, i))
        for j in range(intermediate):
            if i < stop-1:
                s.append( int(s[-1] * np.power(2, (j+1)*1.0/(intermediate+1))))
    return s


def prepare_data(name, ss_size):
    ds = datasets.Datasets().getDataset(name)
    ss_size = ss_size if ss_size < len(ds[0]) else len(ds[0])
    return ds[0][0:ss_size], ds[1][0:ss_size],ds[0][ss_size:-1], ds[1][ss_size:-1]


def run(clf, ds, fold_size):
    start = time.clock()
    err = cross_validation.cross_val_score(clf, ds[0], ds[1], cv=cross_fold)
    elaps = time.clock() - start
    return elaps, err.mean(), err.std()


def run_approx(clf, ds, fold_size, w_size):
    start = time.clock()
    ds_proj = RKSSampler(None, w_size, sigma).transform_cos(ds[0]), ds[1]
    err = cross_validation.cross_val_score(clf, ds_proj[0], ds_proj[1], cv=cross_fold)
    elaps = time.clock() - start
    return elaps, err.mean(), err.std()


def run_2(clf, ds, test):
    start = time.clock()
    clf.fit(ds[0], ds[1])
    elaps = time.clock() - start

    start = time.clock()
    score = clf.score(test[0], test[1])

    return elaps, score, time.clock() - start


def run_approx_2(clf, ds, w_size, test_ds, ref_score=0, sampler=None, gamma=1):
    start = time.clock()
    if sampler is None:
        sampler = RKSSampler(None, w_size, gamma)
    # sampler2 = smp.RBFSampler(gamma=sigma)
    ds_proj = sampler.transform_cos(ds[0]), ds[1]
    clf.fit(ds_proj[0], ds_proj[1])
    elspsed = time.clock() - start

    # start = time.clock()
    # sampler2.fit_transform(test_ds[0])
    # # print sampler2.random_weights_
    # t_ref = time.clock() - start

    start = time.clock()
    features = sampler.transform_cos(test_ds[0])
    time_transform = time.clock() - start
    score = clf.score(features, test_ds[1])
    print 'test appr. w: %6d' % w_size, '\tscore: (%.4f, %.4f)' % (score, ref_score - score), \
        '\t time (tr, trans, d_score):', elspsed, '\t', time_transform, '\t', time.clock() - start - time_transform

    return elspsed, score, time.clock() - start, ref_score - score


def run_approx_2_outliers(clf, ds, w_size, test_ds, outlier1, outlier2, ref_score=0, gamma=1):
    sampler = RKSSampler(None, w_size, sigma)
    res = run_approx_2(clf, ds, w_size, test_ds, ref_score=ref_score, sampler=sampler, gamma=gamma)
    print '\tOutliers test. sigma_3: %.4f \t sigma_4: %.4f' % (clf.score(sampler.transform_cos(outlier1[0]), outlier1[1]),
                                                               clf.score(sampler.transform_cos(outlier2[0]), outlier2[1]))
    return res


def print_res(res_svm, proj_runs, wsizes, ds_name, normalize=False, time_plt=None, erro_plt=None):
    normalization = [res_svm[0]] if normalize else 1
    refere_time = np.array([res_svm[0]]*len(wsizes)) / normalization
    approx_time = np.array([x[0] for x in proj_runs]) / normalization
    time_plt.set_title('relative time')
    time_plt.set_ylabel("time")
    time_plt.loglog(np.array(wsizes), refere_time, lw=2, basey=2)
    time_plt.loglog(np.array(wsizes), approx_time, label=ds_name, basey=2)

    normalization = [res_svm[1]] if normalize else 1
    refere_error = np.array([res_svm[1]]*len(wsizes)) / normalization
    approx_error = np.array([x[1] for x in proj_runs]) / normalization
    erro_plt.set_title('relative score')
    erro_plt.set_ylabel('score')
    erro_plt.semilogx(np.array(wsizes), refere_error)
    erro_plt.semilogx(np.array(wsizes), approx_error, label=ds_name)
    erro_plt.set_xlabel('projection size D')


def savefig(name, postfix=''):
    plt.savefig('./plots/' + name + '_' + postfix + '.png', dpi=300)


def savefig(name, postfix='', fig=None):
    if fig is None:
        fig = plt
    fig.savefig('./plots/' + name + '_' + postfix + '.png', dpi=300)


def log_average(arr):
    arr = np.array(arr)
    np.log(arr, arr)
    res = np.average(arr)
    return np.exp(res)


def aproximate_loglog(x, y, color, label, plot):
    if plot is None:
        plot = plt
    x = np.log(np.array(x))
    y_log = np.log(np.array(y))
    x_vander = np.vander(x, 2)
    res = la.lstsq(x_vander, y_log)[0]
    x = np.linspace(x[0], x[-1])
    x_vander = np.vander(x, 2)
    y = np.dot(x_vander, res)
    x = np.exp(x)
    y = np.exp(y)
    print 'lsqr: ', res, color
    plot.loglog(x, y, '--', lw=1, color=color)


def show_legend(figure):
    figure.legend(fancybox=True, framealpha=0.5, loc='upper left')


# EXPERIMENT 4 Error&Time comparison
def TEST4_timeerror_intervals(N=100000, k=50, drop=5, wmin=50, wmax=1700, intermidiate=1):
    size = N
    full_dataset = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2, n_samples=2**17, random_state=2438)
    # full_dataset = full_dataset[0][0:size], full_dataset[1][0:size]
    ds = full_dataset[0][0:size], full_dataset[1][0:size]

    w_sizes = get_w_sizes(wmin, wmax, intermediate=intermidiate)

    res_t = np.random.rand(len(w_sizes), k)
    res_e = np.random.rand(len(w_sizes), k)
    print 'start'
    # res_svm = [93.019637, 0.9167022705078125, 48.86217599999999]
    res_svm = run_2(svm_clf, ds, full_dataset)
    print 'res_svm', res_svm
    plt.semilogx(res_svm[0], 1 - res_svm[1], 'rs', label='SVM')
    for i in range(k):
        print ''
        print 'RUN', i
        for j, w in enumerate(w_sizes):
            res = run_approx_2(lin_clf, ds, w, full_dataset)  # tt, score, total_t, delta_score
            res_t[j, i] = res[0]
            res_e[j, i] = 1 - res[1]

    print 't', res_t

    res_t = np.sort(res_t, axis=1)[:,drop:-drop]
    res_e = np.sort(res_e, axis=1)[:,drop:-drop]

    print 't2', res_t

    x = np.average(res_t, axis=1)
    y = np.average(res_e, axis=1)
    xerr = res_t[:, [0, -1]]
    yerr = res_e[:, [0, -1]]
    print 'xerr', xerr.T
    print 'yerr', yerr.T
    xerr = np.absolute(xerr - np.array([x, x]).T)
    yerr= np.absolute(yerr - np.array([y, y]).T)
    print 'x', x
    print 'y', y
    print 'xerr', xerr.T
    print 'yerr', yerr.T

    plt.xscale("log")
    plt.errorbar(x, y, xerr=xerr.T, yerr=yerr.T, fmt='o', label='RFF for various D')
    plt.xlim([x[0]*0.7,res_svm[0]*2])

#
# plt.semilogx(res_svm[0], 1-res_svm[1], 'rs')
# for i in range(len(w_sizes)):
#     t = np.sort(res_t[i])[1:-1]
#     e = np.sort(res_e[i])[1:-1]
#     plt.semilogx(t, e, 'bs')

    plt.xlabel('training time')
    plt.ylabel('training error')
    plt.grid()
    plt.title('Running time and error comparison')
    savefig('svm_timeerror_10-90', 'n%dN%dk%dk%d' % (2, N, k, drop))
    plt.legend(fancybox=True, framealpha=0.5)
    plt.show()



#
def TEST4_2_timeerror_intervals(N=100000, k=100, drop=5, wmin=100, wmax=10001, intermidiate=1, d=6, scale=5, gamma=1):
    size = N
    generator = gc.GausianClusters(d, scale)
    full_dataset = generator.generate_classification(N*2)
    ds = full_dataset[0][0:size], full_dataset[1][0:size]

    w_sizes = get_w_sizes(wmin, wmax, intermediate=intermidiate)
    lin_clf = svm.LinearSVC(C=1, loss='hinge')
    svm_clf = svm.SVC(kernel='rbf', C=1, gamma=gamma)

    res_t = np.random.rand(len(w_sizes), k)
    res_e = np.random.rand(len(w_sizes), k)

    print 'start'
    res_svm = run_2(svm_clf, ds, full_dataset)
    print 'res_svm', res_svm
    plt.semilogx(res_svm[0], 1 - res_svm[1], 'rs', label='SVM')
    for i in range(k):
        print ''
        print 'RUN', i
        for j, w in enumerate(w_sizes):
            res = run_approx_2(lin_clf, ds, w, full_dataset)  # tt, score, total_t, delta_score
            res_t[j, i] = res[0]
            res_e[j, i] = 1 - res[1]

    print 't', res_t

    res_t = np.sort(res_t, axis=1)[:,drop:-drop]
    res_e = np.sort(res_e, axis=1)[:,drop:-drop]

    print 't2', res_t

    x = np.average(res_t, axis=1)
    y = np.average(res_e, axis=1)
    xerr = res_t[:, [0, -1]]
    yerr = res_e[:, [0, -1]]
    print 'xerr', xerr.T
    print 'yerr', yerr.T
    xerr = np.absolute(xerr - np.array([x, x]).T)
    yerr= np.absolute(yerr - np.array([y, y]).T)
    print 'x', x
    print 'y', y
    print 'xerr', xerr.T
    print 'yerr', yerr.T

    plt.xscale("log")
    plt.errorbar(x, y, xerr=xerr.T, yerr=yerr.T, fmt='o', label='RFF for various D')
    x = np.append(x, res_svm[0])
    plt.xlim([np.min(x)*0.7, np.max(x)*2])

    plt.xlabel('training time')
    plt.ylabel('validation error')
    plt.grid()

    plt.legend(fancybox=True, framealpha=0.5)
    savefig('svm_timeerror_10-90', 'n%dN%dk%dk%d' % (d, N, k, drop))
    plt.title('Running time and error comparison')
    plt.show()


def score_outliers(cls, set1, set2):
    return cls.score(set1[0], set1[1]), cls.score(set2[0], set2[1])






# EXPERIMENT 6.
# It is an experiment 3, again, but with more complex input dataset
#
# Compare training time and evaluation of linearSVM vs SVM
# Plot the resulting function of time: time(n)
#
# APPROACH:
# Full dataset of max(n) size is generated first.
# Learners are trained on datasets of sizes(2^x, 2^(x+1), ..., 2^y) and error is measured on a dataset double that size
# Approx.learner (linSVM with RFF) are trained for 2 values of error 0.01 and 0.001
# Approx.learner training is performed 'repeats' number of times and resulting times are averaged for visualization.
#   required because fixed number of dimensions for feature space are used and multiple runs (with same settings) might
#   achieve target error on different projection sizes (features are random, what can i do).
# DATASET:
# N dimensional input space
# 2 clusters that represent classification problem
# inputs are drawn according to N-dim normal distribution, scaled randomly
# clusters intersect heavily so some error is guarantied
def TEST6_svm_vs_linsvm_high_d(repeats=5, d=10, scale=5, ds_size=2**17, gamma=None, outlier_size=10000,
                               outlier_gamma_1=2, outlier_gamma_2=2.5, wmin=10, wmax=10001, wintermidiate=2,
                               logNmin=10, logNmax=17, logNintermemdiate=1, logNbase=2):
    print 'started test 6'
    generator = gc.GausianClusters(d, scale)
    full_dataset = generator.generate_classification(ds_size)
    outliers_3 = generator.generate_classification_outliers(outlier_size, outlier_gamma_1)
    outliers_4 = generator.generate_classification_outliers(outlier_size, outlier_gamma_2)

    if gamma is None:
        gamma = gc.GausianClusters.gamma(full_dataset[0])
        print 'gamma', gamma

    lin_clf = svm.LinearSVC(C=1, loss='hinge')
    svm_clf = svm.SVC(kernel='rbf', C=1, gamma=gamma)

    # input_sizes = get_log_sequence(2, 10, 17, intermediate=1)             # LONG
    input_sizes = get_log_sequence(logNbase, logNmin, logNmax, intermediate=logNintermemdiate)
    w_size = get_w_sizes(wmin, wmax, intermediate=wintermidiate)

    print 'input sizes', input_sizes

    time_90, time_99 = [], []
    tt_svm, tt_90, tt_99 = [], [], []
    time_svm = []

    print input_sizes

    for index_i, input_size in enumerate(input_sizes):
        ds = full_dataset[0][0:input_size], full_dataset[1][0:input_size]
        test = full_dataset[0][0:input_size*2], full_dataset[1][0:input_size*2]

        res_svm = run_2(svm_clf, ds, test)    #t, sc, eval
        print 'INPUT', input_size, 'SVM. t: %.6f \t s: %.4f \t tt: %.6f' % res_svm + ' \t |SV|:%d' % svm_clf.support_vectors_.shape[0]
        time_svm.append(res_svm[0])
        tt_svm.append(res_svm[2])

        print 'Outliers test. sigma_3: %.6f \t sigma_4: %.6f' % score_outliers(svm_clf, outliers_3, outliers_4)

        w_90, w_t_90, w_99, w_t_99 = [], [], [], []
        for k in range(repeats):
            for index_j,  w in enumerate(w_size):
                res = run_approx_2_outliers(lin_clf, ds, w, test, outliers_3, outliers_4, gamma=gamma, ref_score=res_svm[1])  # tt, score, total_t, delta_score
                if res[3] <= 0.01 and len(w_90) == len(w_99):
                    w_90.append(res[0])
                    w_t_90.append(res[2])
                if res[3] <= 0.001:
                    w_99.append(res[0])
                    w_t_99.append(res[2])
                    break
                if index_j == len(w_size)-1:
                    print 'not happened!'
        print w_90, log_average(w_90)
        print w_99, log_average(w_99)
        time_90.append(log_average(w_90))
        time_99.append(log_average(w_99))
        tt_90.append(log_average(w_t_90))
        tt_99.append(log_average(w_t_99))
        print

    # test6_print_routine_together(d, ds_size, gamma, generator, input_sizes, repeats,
    #                              time_90, time_99, time_svm, tt_90, tt_99)

    test6_print_routine(d, ds_size, gamma, generator, input_sizes, repeats, time_90, time_99, time_svm, tt_90, tt_99)


def test6_print_routine_together(d, ds_size, gamma, generator, input_sizes, repeats, time_90, time_99, time_svm, tt_90,
                                 tt_99):
    # OUTPUT
    fig = plt.figure()
    DefaultSize = fig.get_size_inches()
    size_mult = 1.5
    fig.set_size_inches((DefaultSize[0] * size_mult, DefaultSize[1] * size_mult))
    fig.suptitle('d: %d; gamma: %.4f; repeats: %d' % (d, gamma, repeats), fontsize=14, fontweight='bold')
    x = np.array(input_sizes)

    # LOGLOG plot. nice, but not as sharp as linear
    figure = plt.subplot(223)
    f1 = figure
    figure.loglog(x, time_svm, lw=2, color='y', label="SVM")
    figure.loglog(x, time_90, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.loglog(x, time_99, lw=2, color='r', label="RFF delta(e) <= 0.001")
    aproximate_loglog(input_sizes, time_90, color='b', label='Approx for delta(e) <= 0.01', plot=figure)
    aproximate_loglog(input_sizes, time_99, color='r', label='Approx for delta(e) <= 0.001', plot=figure)
    figure.set_xlabel('input size')
    figure.set_ylabel('training time')
    figure.grid()

    # training time comparison
    figure = plt.subplot(224)
    f2 = figure
    figure.plot(x, time_svm, lw=2, color='y', label="SVM")
    figure.plot(x, time_90, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.plot(x, time_99, lw=2, color='r', label="RFF delta(e) <= 0.001")
    figure.set_xlabel('input size')
    figure.set_ylabel('training time')
    figure.grid()

    # evaluation time comparison
    figure = plt.subplot(222)
    f3 = figure
    figure.loglog(x, time_svm / x, lw=2, color='y', label="SVM")
    figure.loglog(x, tt_90 / x, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.loglog(x, tt_99 / x, lw=2, color='r', label="RFF delta(e) <= 0.001")
    # Least squares for eval.time. Looks misleading
    # aproximate_loglog(input_sizes, tt_90/x, color='b', label='Approx for delta(e) <= 0.01', plot=figure)
    # aproximate_loglog(input_sizes, tt_99/x, color='r', label='Approx for delta(e) <= 0.001', plot = figure)
    # aproximate_loglog(input_sizes, time_svm/x, color='y', label='Approx for delta(e) <= 0.001', plot = figure)
    figure.set_xlabel('input size')
    figure.set_ylabel('training time')
    figure.grid()

    # inputs
    example_size = 1000
    x, y = generator.generate_classification(example_size)
    x = PCA(x).Y
    ax = plt.subplot(221)
    # 3D input plot. Looks not that great
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

    ds = x, y
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
    show_legend(f1)
    show_legend(f2)
    show_legend(f3)
    savefig('svm_tm_n%d_N%d_k%d--Input' % (d, ds_size, repeats))
    ax.set_title("Classification problem. 1500 input points.")
    f1.set_title('Training time of SVM and linear SMV using RFF')
    f2.set_title('Training time of SVM and linear SMV using RFF')
    f3.set_title('Comparison of evaluation time of SVM and linear SMV using RFF')


def test6_print_routine(d, ds_size, gamma, generator, input_sizes, repeats, time_90, time_99, time_svm, tt_90, tt_99):
    fig = plt.figure()
    # DefaultSize = fig.get_size_inches()
    # size_mult = 1.5
    # fig.set_size_inches((DefaultSize[0] * size_mult, DefaultSize[1] * size_mult))
    # fig.suptitle('d: %d; gamma: %.4f; repeats: %d' % (d, gamma, repeats), fontsize=14, fontweight='bold')

    x = np.array(input_sizes)

    # LOGLOG plot. nice, but not as sharp as linear
    plt.figure()
    figure = plt.subplot(111)
    f1 = figure
    figure.loglog(x, time_svm, lw=2, color='y', label="SVM")
    figure.loglog(x, time_90, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.loglog(x, time_99, lw=2, color='r', label="RFF delta(e) <= 0.001")
    aproximate_loglog(input_sizes, time_90, color='b', label='Approx for delta(e) <= 0.01', plot=figure)
    aproximate_loglog(input_sizes, time_99, color='r', label='Approx for delta(e) <= 0.001', plot=figure)
    figure.set_xlabel('input size')
    figure.set_ylabel('training time')
    figure.grid()
    show_legend(figure)
    savefig('svm_tm_n%d_N%d_k%d-lolo' % (d, ds_size, repeats))
    figure.set_title('Training time of SVM and linear SMV using RFF')


    # training time comparison
    plt.figure()
    figure = plt.subplot(111)
    f2 =figure
    figure.plot(x, time_svm, lw=2, color='y', label="SVM")
    figure.plot(x, time_90, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.plot(x, time_99, lw=2, color='r', label="RFF delta(e) <= 0.001")
    figure.set_xlabel('input size')
    figure.set_ylabel('training time')
    figure.grid()
    show_legend(figure)

    savefig('svm_tm_n%d_N%d_k%d--lin' % (d, ds_size, repeats))
    figure.set_title('Training time of SVM and linear SMV using RFF')

    # evaluation time comparison
    plt.figure()
    figure = plt.subplot(111)
    figure.loglog(x, time_svm / x, lw=2, color='y', label="SVM")
    figure.loglog(x, tt_90 / x, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.loglog(x, tt_99 / x, lw=2, color='r', label="RFF delta(e) <= 0.001")
    # Least squares for eval.time. Looks misleading
    # aproximate_loglog(input_sizes, tt_90/x, color='b', label='Approx for delta(e) <= 0.01', plot=figure)
    # aproximate_loglog(input_sizes, tt_99/x, color='r', label='Approx for delta(e) <= 0.001', plot = figure)
    # aproximate_loglog(input_sizes, time_svm/x, color='y', label='Approx for delta(e) <= 0.001', plot = figure)
    figure.set_xlabel('input size')
    figure.set_ylabel('training time')
    figure.grid()
    show_legend(figure)
    savefig('svm_tm_n%d_N%d_k%d--eval' % (d, ds_size, repeats))
    figure.set_title('Comparison of evaluation time of SVM and linear SMV using RFF')

    # inputs
    # 3D input plot. Looks not that great
    # show inputs
    x1, x2 = generator.generate(500)
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    marker = '.'
    xp, yp, zp = gc.GausianClusters.pca(x1)
    ax.scatter(xp, yp, zp, c='r', marker=marker)
    xp, yp, zp = gc.GausianClusters.pca(x2)
    ax.scatter(xp, yp, zp, c='b', marker=marker)
    # print box to maintain the scales
    xp, yp, zp = generator.box_3d()
    ax.scatter(xp, yp, zp, '.', c='white', alpha=0.0)
    savefig('svm_tm_n%d_N%d_k%d--3d' % (d, ds_size, repeats))

    example_size = 1000
    x, y = generator.generate_classification(example_size)
    x = PCA(x).Y
    ax = plt.subplot(111)
    ds = x, y
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
    show_legend(f1)
    show_legend(f2)

    savefig('svm_tm_n%d_N%d_k%d--2d' % (d, ds_size, repeats))
    ax.set_title("Classification problem. 1500 input points.")


# EXPERIMENT 7.
# Plot: required D-size (projection) depending on the dimensionality of input data (d)
#
# Generate related input familites for d= dmin, dmin+1, ..., dmax
# Run REPEATS test on each input family and measure time and avg.projection size required to achieve accuracy
def TEST7_convergence(repeats=10, dmin=1, dmax=100, scale=5, N=10000, gamma=1, wmin=10, wmax=100001, wintermidiate=3):
    print 'started test 7. Lets figue out convergence rate'
    # if gamma is None:
    #     gamma = gc.GausianClusters.gamma(full_dataset[0])
    #     print gamma, gamma

    lin_clf = svm.LinearSVC(C=1, loss='hinge')
    svm_clf = svm.SVC(kernel='rbf', C=1, gamma=gamma)

    w_size = get_w_sizes(wmin, wmax, intermediate=wintermidiate)

    # we want datasets to be alike
    seed = gc.GausianClusters(dmax, scale)

    dim = []
    m_90, m_99 = [], []
    t_90, t_99 = [], []

    for d in range(dmin, dmax):
        print '\n\r\n\r D:', d
        generator = gc.GausianClusters(d, scale, base=seed)
        full_dataset = generator.generate_classification(N*2)

        ds = full_dataset[0][0:N], full_dataset[1][0:N]
        test = full_dataset

        res_svm = run_2(svm_clf, ds, test)    #t, sc, eval
        ref_score = res_svm[1]
        print 'INPUT', 'SVM. t: %.6f \t s: %.4f \t tt: %.6f' % res_svm + ' \t |SV|:%d' % svm_clf.support_vectors_.shape[0]

        print 'Outliers test. sigma_3: %.6f \t sigma_4: %.6f'

        km_90, kt_90, km_99, kt_99 = [], [], [], []
        for k in range(repeats):
            for index_j, w in enumerate(w_size):
                res = run_approx_2(lin_clf, ds, w, test, gamma=gamma, ref_score=res_svm[1])  # tt, score, total_t, delta_score
                if res[3] <= 0.01 and len(km_90) == len(km_99):
                    km_90.append(w)
                    kt_90.append(res[2])
                if res[3] <= 0.001:
                    km_99.append(w)
                    kt_99.append(res[2])
                    break
                if index_j == len(w_size)-1:
                    print 'not happened!'

        dim.append(d)
        m_90.append(np.array(km_90).mean())
        m_99.append(np.array(km_99).mean())
        t_90.append(np.array(kt_90).mean())
        t_99.append(np.array(kt_99).mean())
        print 'kt_90', kt_90
        print 'km_90', km_90
        print 't_90', t_90
        print 'm_90', m_90

        print

    figure = plt.subplot(211)
    f1 = figure
    figure.plot(dim, m_90, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.plot(dim, m_99, lw=2, color='r', label="RFF delta(e) <= 0.001")
    figure.set_xlabel('input dimensionality')
    figure.set_ylabel('avg.projection dimensionality')
    figure.grid()
    figure = plt.subplot(212)
    f2 = figure
    figure.plot(dim, t_90, lw=2, color='b', label="RFF delta(e) <= 0.01")
    figure.plot(dim, t_99, lw=2, color='r', label="RFF delta(e) <= 0.001")
    figure.set_xlabel('input dimensionality')
    figure.set_ylabel('avg. approximation training time')
    figure.grid()
    savefig('rff_N%d_D%d-%d_k%d' % (N, dmin, dmax, repeats))


def draw_input_space():
    """Draw 2D plot with 2 intersecting 2d gausian clusters"""
    for i in range(1):
        # state = [9448, 7336, 8418, 8668, 9555, 194, 4246][i]
        state = rnd.randint(1, 10000)
        ds = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2, n_samples=2**17, random_state=2438)
        x1, x2, y1, y2 = [], [], [], []
        for i in range(1500):
            if ds[1][i] == 1:
                x1.append(ds[0][i][0])
                y1.append(ds[0][i][1])
            else:
                x2.append(ds[0][i][0])
                y2.append(ds[0][i][1])

        plt.plot(x1, y1, 'bs')
        plt.plot(x2, y2, 'r^')
        plt.title("Classification problem. 1500 input points.")
        plt.show()


# EXPERIMENT 1
# Uses illnesses datasets and checks the performance using different projection sizes
cross_fold = 4
sigma = 1


def TEST1_compare_on_illnesses(subset_size=200, intermidiate_w_sizes=4):
    ds_names = datasets.Datasets().getAllDatasetNames()
    print ds_names
    lin_clf = svm.LinearSVC(C=1, loss='hinge')
    svm_clf = svm.SVC(kernel='rbf', C=1, gamma=sigma)
    projection_sizes = get_w_sizes(10, 10000, intermediate=intermidiate_w_sizes)
    f, (time_plt, erro_plt) = plt.subplots(2, sharex='col')

    for ds_name in [y for y in ds_names if y != 'magic04' and y != 'Nomao']:
        try:
            print ds_name
            ds = prepare_data(ds_name, subset_size)
            print ds[1]
            if len(ds[0]) < subset_size:
                print ds_name, 'is not big enough'
                continue
            res_svm = run(svm_clf, ds, cross_fold)

            proj_runs = []
            for w in projection_sizes:
                proj_runs.append(run_approx(lin_clf, ds, cross_fold, w))
            print_res(res_svm, proj_runs, projection_sizes, ds_name, True, time_plt, erro_plt)
        except:
            print 'ds %s was uncool' % ds_name
            print sys.exc_info()[0]
            raise

    savefig("svm_vs_lin", str(subset_size))
    plt.show()


# EXPERIMENT 2 - 0.01, 0.001 error
# Just generate some stuff and run for bunch of input sizes and projection sizes.
# And compare it to SVM.
# this is official.
def TEST2_run_for_banch_of_input_size():
    full_dataset = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2, n_samples=2**17, random_state=2438)

    input_sizes = get_log_sequence(2, 4, 16, intermediate=1)
    w_size = get_w_sizes(10, 5000, intermediate=2)

    time_90 = []
    time_95 = []
    time_99 = []

    lin_clf = svm.LinearSVC(C=1, loss='hinge')
    svm_clf = svm.SVC(kernel='rbf', C=1, gamma=sigma)

    wtimes = np.zeros((len(w_size), len(input_sizes)), np.float)
    werror = np.zeros((len(w_size), len(input_sizes)), np.float)

    print input_sizes

    for index_i, input_size in enumerate(input_sizes):
        ds = full_dataset[0][0:input_size], full_dataset[1][0:input_size]
        test = full_dataset[0][0:input_size*2], full_dataset[1][0:input_size*2]

        res_svm = run_2(svm_clf, ds, test)    #t, sc

        proj_runs = []
        approx_eval_time = None
        found = False
        for index_j, w in enumerate(w_size):
            proj_runs.append(run_approx_2(lin_clf, ds, w, test))
            wtimes[index_j][index_i] = proj_runs[-1][0] / res_svm[0]
            approx_eval_time = proj_runs[-1][2]
            if res_svm[1] - proj_runs[-1][1] < 0.001:
                found = True
            # werror = proj_runs[-1][1] / res_svm[1]
        if not found:
            proj_runs.append(run_approx_2(lin_clf, ds, w_size[-1]*3, test))

        score_list = (np.array([x[1] for x in proj_runs]))
        # relative_error_list = np.absolute(score_list - 1)/(1-res_svm[1] if res_svm[1] != 1 else 0.01)
        relative_error_list = (score_list - res_svm[1]) * -1
        print 'reror', relative_error_list
        relative_time_proj = np.array([x[0] for x in proj_runs]) / res_svm[0]

        index_90, index_95, index_99 = None, None, None
        for i in range(len(score_list)):
            if relative_error_list[i] <= 0.01 and index_90 is None:
                index_90 = i
            if relative_error_list[i] <= 0.005 and index_95 is None:
                index_95 = i
            if relative_error_list[i] <= 0.001 and index_99 is None:
                index_99 = i
        time_90.append(relative_time_proj[index_90] if index_90 is not None else None)
        time_95.append(relative_time_proj[index_95] if index_95 is not None else None)
        time_99.append(relative_time_proj[index_99] if index_99 is not None else 0)
        print 'TIME. SVM:', res_svm[0], '\tproj relative:', relative_time_proj[-1], '\tproj worse:', proj_runs[-1][0]
        print "input_size", input_size, '\t\tscore:', res_svm[1], '\tindexes', index_90, index_95, index_99
        print 'error svm:', res_svm[1], "runs", relative_error_list[index_90], relative_error_list[index_95], relative_error_list[index_99]
        print 'eval (svm, proj):', res_svm[2], approx_eval_time
        print

    plt.figure()

    # plot supporting lines with for every projection size
    # for i in range(len(w_size)):
    #     plt.loglog(np.array(input_sizes), wtimes[i], lw=1, basey=2, color='#cccccc')

    print 'time 90', time_90, time_99
    plt.loglog(np.array(input_sizes), np.ones(len(input_sizes)), lw=2, basey=2, color='y', label="SVM (const 1)")
    plt.loglog(np.array(input_sizes), time_90, lw=2, basey=2, color='b', label="RFF delta(e) <= 0.01")
    # plt.loglog(np.array(input_sizes), time_95, lw=2, basey=2, color='y', label="0.05")
    plt.loglog(np.array(input_sizes), time_99, lw=2, basey=2, color='r', label="RFF delta(e) <= 0.001")
    plt.xlabel('input size')
    plt.ylabel('relative training time')
    savefig('earlyandugly')
    plt.title('Comparison of training time of SVM and linear SMV using RFF')
    plt.legend()


def run_relevant_tests():
    # TEST1_compare_on_illnesses()
    # plt.figure()
    TEST2_run_for_banch_of_input_size()
    plt.figure()
    TEST4_2_timeerror_intervals(N=20000, k=100, drop=5)
    plt.figure()
    TEST6_svm_vs_linsvm_high_d(repeats=10)
    plt.show()


def run_forest_runasfastasyoucan():
    TEST4_timeerror_intervals(N=100, k=5, drop=1)
    plt.figure()
    TEST6_svm_vs_linsvm_high_d(logNmax=12, d=5, repeats=3)
    plt.show()

run_relevant_tests()

# TEST2_run_for_banch_of_input_size()
# plt.figure()
# TEST1_compare_on_illnesses()
# TEST4_2_timeerror_intervals(N=20000, k=100, drop=5)
# plt.figure()
# TEST6_svm_vs_linsvm_high_d(repeats=15)
# TEST4_timeerror_intervals()       #full
# TEST4_timeerror_intervals(N=100, k=5, drop=1) #fast
# TEST4_2_timeerror_intervals(N=20000)
# TEST4_2_timeerror_intervals(N=100, k=5, drop=1, d=6, scale=5)

# TEST7_convergence(repeats=10, dmin=10, dmax=60, scale=1, N=10000)
# plt.show()
#
# TEST6_svm_vs_linsvm_high_d()
# # TEST6_svm_vs_linsvm_high_d(logNmax=12, d=5, repeats=3)
#
# plt.figure()
#
# # TEST4_timeerror_intervals()       #full
# # TEST4_timeerror_intervals(N=100, k=5, drop=1) #fast
# TEST4_2_timeerror_intervals(N=10000)
# # TEST4_2_timeerror_intervals(N=100, k=5, drop=1, d=6, scale=5)

# TEST7_convergence(repeats=10, dmin=10, dmax=60, scale=1, N=10000)