import datasets
import numpy as np
import time
import sys
from sklearn import datasets as ds
from sklearn import svm
from sklearn import cross_validation
from RKS_Sampler import RKSSampler
from matplotlib import pyplot as plt


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
        for j in range(intermediate):
            s.append(int(i * np.power(10, (j+1)*1.0/(intermediate+1))))
        i *= 10
    s.append(i)
    print s
    return s


def get_log_sequence(base, start, stop):
    s = []
    for i in range(start, stop):
        s.append(np.power(base, i))
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


def run_2(clf, ds):
    start = time.clock()
    clf.fit(ds[0], ds[1])
    elaps = time.clock() - start
    return elaps, clf.score(ds[0], ds[1])


def run_approx_2(clf, ds, w_size):
    start = time.clock()
    ds_proj = RKSSampler(None, w_size, sigma).transform_cos(ds[0]), ds[1]
    clf.fit(ds_proj[0], ds_proj[1])
    elaps = time.clock() - start
    return elaps, clf.score(ds_proj[0], ds[1])


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
projection_sizes = get_w_sizes(intermediate=intermediate_sizes)



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

input_sizes = get_log_sequence(2, 3, 15)
w_size = get_w_sizes(10, 1000)

better_t = []
better_e = []
worse_t = []
worse_e = []
for input_size in input_sizes:
    ds = make_classification(n_features=6, n_redundant=0, n_informative=6, n_clusters_per_class=3, n_samples=input_size)
    res_svm = run_2(svm_clf, ds)    #t, sc

    proj_runs = []
    for w in w_size:
        proj_runs.append(run_approx_2(lin_clf, ds, w))

    score_list = np.array([x[1] for x in proj_runs]) / res_svm[1]
    time_list = np.array([x[0] for x in proj_runs]) / res_svm[0]
    low, gre = 0, len(score_list) - 1
    for i in range(len(score_list)):
        if score_list[i] < 1 and score_list[i] >= score_list[low]:
            low = i
        if score_list[i] > 1 and score_list[i] <= score_list[gre]:
            gre = i
    better_e.append(score_list[gre])
    better_t.append(time_list[gre])
    worse_e.append(score_list[low])
    worse_t.append(time_list[low])
    print res_svm[0], time_list
    print "input_size", input_size, res_svm


plt.loglog(np.array(input_sizes), better_t, lw=2, basey=2,  color='b', label="better_t")
plt.loglog(np.array(input_sizes), better_e, lw=2, basey=2,  color='g',  label="better_e")
plt.loglog(np.array(input_sizes), worse_t, lw=2, basey=2, color='r', label="worse_t")
plt.loglog(np.array(input_sizes), worse_e, lw=2, basey=2, color='y', label="worse_e")
plt.legend()
plt.show()
