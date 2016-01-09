import datasets
import numpy as np
import time
import sys
from sklearn import datasets as ds
from sklearn import svm
from sklearn import cross_validation
from RKS_Sampler import RKSSampler
from matplotlib import pyplot as plt
import random as rnd


from sklearn.datasets import make_classification


# [ 'breast-w', 'colic',    'credit-a', 'credit-g', 'heart-statlog',    'hepatitis',    'magic04',  'sick',    'vote']
#   (699, 9),   (368, 7),   (690, 6),   (1000, 7),  (270, 13),          (155, 19),      (19019, 9), (3772, 7), (435, 16

test_data_home = "Data"
leuk = ds.fetch_mldata('leukemia', transpose_data=True)
leuk = ds.load_digits()


#
# svm_clf2 = svm.SVC(kernel='rbf', C=1, gamma=1.0)
# svm_clf2.fit(leuk[0], leuk[1])
# print leuk[0].shape, leuk[1].shape
# print "score", svm_clf2.score(leuk[0], leuk[1]), "--=-90-90-9"
# svm_clf2 = svm.LinearSVC(C=1, loss='hinge')
# svm_clf2.fit(leuk[0], leuk[1])
# print leuk[0].shape, leuk[1].shape
# print "score", svm_clf2.score(leuk[0], leuk[1])

def get_w_sizes(min_v=10, max_v=10000, intermediate=0):
    i = min_v
    s = []
    print s
    while i < max_v:
        s.append(i)
        print i, max_v
        for j in range(intermediate):
            if (i * np.power(10, (j+1)*1.0/(intermediate+1))) < max_v:
                print i * np.power(10, (j+1)*1.0/(intermediate+1)), max_v
                val = int(i * np.power(10, (j+1)*1.0/(intermediate+1)))
                if val < max_v:
                    s.append(val)
                else:
                    print 'illigal wsize: ', val
        i *= 10
    print s
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


def run_approx_2(clf, ds, w_size, test_ds):
    start = time.clock()
    sampler = RKSSampler(None, w_size, sigma)
    ds_proj = sampler.transform_cos(ds[0]), ds[1]
    clf.fit(ds_proj[0], ds_proj[1])
    elspsed = time.clock() - start

    start = time.clock()
    features = sampler.transform_cos(test_ds[0])
    time_transform = time.clock() - start
    score = clf.score(features, test_ds[1])
    print 'test: %6d' % w_size, '\tscore: %.4f' % score, '\t time (transform, score):', time_transform, '\t', time.clock() - start
    return elspsed, score, time.clock() - start


def print_res(res_svm, proj_runs, wsizes, ds_name, normalize=False):
    normalization = [res_svm[0]] if normalize else 1
    refere_time = np.array([res_svm[0]]*len(wsizes)) / normalization
    approx_time = np.array([x[0] for x in proj_runs]) / normalization
    time_plt.set_title('relative time')
    time_plt.set_ylabel("t")
    time_plt.loglog(np.array(wsizes), refere_time, lw=2, basey=2)
    time_plt.loglog(np.array(wsizes), approx_time, label=ds_name, basey=2)

    normalization = [res_svm[1]] if normalize else 1
    refere_error = np.array([res_svm[1]]*len(wsizes)) / normalization
    approx_error = np.array([x[1] for x in proj_runs]) / normalization
    erro_plt.set_title('relative score')
    erro_plt.set_ylabel('1-e')
    erro_plt.semilogx(np.array(wsizes), refere_error)
    erro_plt.semilogx(np.array(wsizes), approx_error, label=ds_name)

    # print ds_name, res_svm, proj_runs
    if res_svm[2] == 0:
        return
    normalization = [res_svm[2]] if normalize else 1
    refere_gamma = np.array([res_svm[2]]*len(wsizes)) / normalization
    approx_gamma = np.array([x[2] for x in proj_runs]) / normalization
    gamm_plt.set_title('relative error deviation')
    gamm_plt.set_ylabel('gamma')
    gamm_plt.set_xlabel('projection sizes')
    gamm_plt.loglog(np.array(wsizes), refere_gamma, basey=2)
    gamm_plt.loglog(np.array(wsizes), approx_gamma, label=ds_name, basey=2)


def savefig(name, postfix=''):
    plt.savefig('./plots/' + name + '_' + postfix + '.png')


subset_size = 100
cross_fold = 4
sigma = 1
ds_names = datasets.Datasets().getAllDatasetNames()
# ds_names = ['colic']
intermediate_sizes = 4

lin_clf = svm.LinearSVC(C=1, loss='hinge')
svm_clf = svm.SVC(kernel='rbf', C=1, gamma=sigma)

sets = datasets.Datasets()
# projection_sizes = get_w_sizes(intermediate=intermediate_sizes)


# EXPERIMENT 3

full_dataset = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2, n_samples=2**17, random_state=2438)

input_sizes = get_log_sequence(2, 10, 13, intermediate=1)
w_size = get_w_sizes(10, 5000, intermediate=2)

time_90 = []
time_99 = []
time_svm = []

print input_sizes

for index_i, input_size in enumerate(input_sizes):
    ds = full_dataset[0][0:input_size], full_dataset[1][0:input_size]
    test = full_dataset[0][0:input_size*2], full_dataset[1][0:input_size*2]

    res_svm = run_2(svm_clf, ds, test)    #t, sc
    time_svm.append(res_svm[0])

    proj_runs = []
    approx_eval_time = None
    found = False
    for index_j, w in enumerate(w_size):
        proj_runs.append(run_approx_2(lin_clf, ds, w, test))
        approx_eval_time = proj_runs[-1][2]
        if res_svm[1] - proj_runs[-1][1] < 0.001:
            found = True
    if not found:
        proj_runs.append(run_approx_2(lin_clf, ds, w_size[-1]*3, test))

    score_list = (np.array([x[1] for x in proj_runs]))
    relative_error_list = (score_list - res_svm[1]) * -1
    print 'reror', relative_error_list
    relative_time_proj = np.array([x[0] for x in proj_runs])

    index_90, index_99 = None, None
    for i in range(len(score_list)):
        if relative_error_list[i] <= 0.01 and index_90 is None:
            index_90 = i
        if relative_error_list[i] <= 0.001 and index_99 is None:
            index_99 = i
    time_90.append(relative_time_proj[index_90] if index_90 is not None else None)
    time_99.append(relative_time_proj[index_99] if index_99 is not None else 0)
    print 'TIME. SVM:', res_svm[0], '\tproj relative:', relative_time_proj[-1], '\tproj worse:', proj_runs[-1][0]
    print "input_size", input_size, '\t\tscore:', res_svm[1], '\tindexes', index_90, index_99
    print 'error svm:', res_svm[1], "runs", relative_error_list[index_90], relative_error_list[index_99]
    print 'eval (svm, proj):', res_svm[2], approx_eval_time
    print

plt.figure()

print 'time 90', time_90, time_99
plt.loglog(np.array(input_sizes), time_svm, lw=2, basey=2, color='g', label="SVM")
plt.loglog(np.array(input_sizes), time_90, lw=2, basey=2, color='b', label="RFF delta(e) <= 0.01")
plt.loglog(np.array(input_sizes), time_99, lw=2, basey=2, color='r', label="RFF delta(e) <= 0.001")
plt.xlabel('input size')
plt.ylabel('relative training time')
plt.title('Comparison of training time of SVM and linear SMV using RFF')
plt.legend()
plt.show()


# EXPERIMENT 1
# f, (time_plt, erro_plt, gamm_plt) = plt.subplots(3, sharex='col')
# f.set_size_inches(f.get_size_inches()*2)


# for ds_name in [y for y in ds_names if y != 'magic04']:
#     try:
#         print ds_name
#         ds = prepare_data(ds_name, subset_size)
#         print ds[1]
#         if len(ds[0]) < subset_size:
#             print ds_name, 'is not big enough'
#             continue
#         res_svm = run(svm_clf, ds, cross_fold)
#
#         proj_runs = []
#         for w in projection_sizes:
#             proj_runs.append(run_approx(lin_clf, ds, cross_fold, w))
#         print_res(res_svm, proj_runs, projection_sizes, ds_name, True)
#     except:
#         print 'ds %s was uncool' % ds_name
#         print sys.exc_info()[0]
#         raise
#
# savefig("svm_vs_lin", str(subset_size))
# plt.show()

# # # # DRAW iNPUT SPACE
# for i in range(1):
#     # state = [9448, 7336, 8418, 8668, 9555, 194, 4246][i]
#     state = rnd.randint(1, 10000)
#     ds = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2, n_samples=2**17, random_state=2438)
#     x1, x2, y1, y2 = [], [], [], []
#     for i in range(1500):
#         if ds[1][i] == 1:
#             x1.append(ds[0][i][0])
#             y1.append(ds[0][i][1])
#         else:
#             x2.append(ds[0][i][0])
#             y2.append(ds[0][i][1])
#
#     plt.plot(x1, y1, 'bs')
#     plt.plot(x2, y2, 'r^')
#     plt.title("Classification problem. 1500 input points.")
#     plt.show()
#


# EXPERIMENT 2 - 0.01, 0.001 error
# full_dataset = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=2, n_samples=2**17, random_state=2438)
#
# input_sizes = get_log_sequence(2, 4, 16, intermediate=1)
# w_size = get_w_sizes(10, 5000, intermediate=2)
#
# time_90 = []
# time_95 = []
# time_99 = []
#
# wtimes = np.zeros((len(w_size), len(input_sizes)), np.float)
# werror = np.zeros((len(w_size), len(input_sizes)), np.float)
#
# print input_sizes
#
# for index_i, input_size in enumerate(input_sizes):
#     ds = full_dataset[0][0:input_size], full_dataset[1][0:input_size]
#     test = full_dataset[0][0:input_size*2], full_dataset[1][0:input_size*2]
#
#     res_svm = run_2(svm_clf, ds, test)    #t, sc
#
#     proj_runs = []
#     approx_eval_time = None
#     found = False
#     for index_j, w in enumerate(w_size):
#         proj_runs.append(run_approx_2(lin_clf, ds, w, test))
#         wtimes[index_j][index_i] = proj_runs[-1][0] / res_svm[0]
#         approx_eval_time = proj_runs[-1][2]
#         if res_svm[1] - proj_runs[-1][1] < 0.001:
#             found = True
#         # werror = proj_runs[-1][1] / res_svm[1]
#     if not found:
#         proj_runs.append(run_approx_2(lin_clf, ds, w_size[-1]*3, test))
#
#     score_list = (np.array([x[1] for x in proj_runs]))
#     # relative_error_list = np.absolute(score_list - 1)/(1-res_svm[1] if res_svm[1] != 1 else 0.01)
#     relative_error_list = (score_list - res_svm[1]) * -1
#     print 'reror', relative_error_list
#     relative_time_proj = np.array([x[0] for x in proj_runs]) / res_svm[0]
#
#     index_90, index_95, index_99 = None, None, None
#     for i in range(len(score_list)):
#         if relative_error_list[i] <= 0.01 and index_90 is None:
#             index_90 = i
#         if relative_error_list[i] <= 0.005 and index_95 is None:
#             index_95 = i
#         if relative_error_list[i] <= 0.001 and index_99 is None:
#             index_99 = i
#     time_90.append(relative_time_proj[index_90] if index_90 is not None else None)
#     time_95.append(relative_time_proj[index_95] if index_95 is not None else None)
#     time_99.append(relative_time_proj[index_99] if index_99 is not None else 0)
#     print 'TIME. SVM:', res_svm[0], '\tproj relative:', relative_time_proj[-1], '\tproj worse:', proj_runs[-1][0]
#     print "input_size", input_size, '\t\tscore:', res_svm[1], '\tindexes', index_90, index_95, index_99
#     print 'error svm:', res_svm[1], "runs", relative_error_list[index_90], relative_error_list[index_95], relative_error_list[index_99]
#     print 'eval (svm, proj):', res_svm[2], approx_eval_time
#     print
#
# plt.figure()
#
# for i in range(len(w_size)):
#     plt.loglog(np.array(input_sizes), wtimes[i], lw=1, basey=2, color='#cccccc')
#
# print 'time 90', time_90, time_99
# plt.loglog(np.array(input_sizes), np.ones(len(input_sizes)), lw=2, basey=2, color='y', label="SVM (const 1)")
# plt.loglog(np.array(input_sizes), time_90, lw=2, basey=2, color='b', label="RFF delta(e) <= 0.01")
# # plt.loglog(np.array(input_sizes), time_95, lw=2, basey=2, color='y', label="0.05")
# plt.loglog(np.array(input_sizes), time_99, lw=2, basey=2, color='r', label="RFF delta(e) <= 0.001")
# plt.xlabel('input size')
# plt.ylabel('relative training time')
# plt.title('Comparison of training time of SVM and linear SMV using RFF')
# plt.legend()
# plt.show()

