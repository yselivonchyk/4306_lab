import datasets
import numpy as np
import time
from sklearn import svm
from sklearn import cross_validation
from RKS_Sampler import RKSSampler
from matplotlib import pyplot as plt

# [ 'breast-w', 'colic',    'credit-a', 'credit-g', 'heart-statlog',    'hepatitis',    'magic04',  'sick',    'vote']
#   (699, 9),   (368, 7),   (690, 6),   (1000, 7),  (270, 13),          (155, 19),      (19019, 9), (3772, 7), (435, 16


def get_w_sizes(min_v=10, max_v=1000, intermediate=0):
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


f, (time_plt, erro_plt, gamm_plt) = plt.subplots(3, sharex='col')


def print_res(res_svm, proj_runs, wsizes, ds_name, normalize=False):
    normalization = [res_svm[0]] if normalize else 1
    refere_time = np.array([res_svm[0]]*len(wsizes)) / normalization
    approx_time = np.array([x[0] for x in proj_runs]) / normalization
    time_plt.set_title('relative time')
    time_plt.set_ylabel("t")
    time_plt.loglog(np.array(wsizes), refere_time)
    time_plt.loglog(np.array(wsizes), approx_time, label=ds_name)

    normalization = [res_svm[1]] if normalize else 1
    refere_error = np.array([res_svm[1]]*len(wsizes)) / normalization
    approx_error = np.array([x[1] for x in proj_runs]) / normalization
    erro_plt.set_title('relative error')
    erro_plt.set_ylabel('e')
    erro_plt.semilogx(np.array(wsizes), refere_error)
    erro_plt.semilogx(np.array(wsizes), approx_error, label=ds_name)

    normalization = [res_svm[2]] if normalize else 1
    refere_gamma = np.array([res_svm[2]]*len(wsizes)) / normalization
    approx_gamma = np.array([x[2] for x in proj_runs]) / normalization
    gamm_plt.set_title('relative error deviation')
    gamm_plt.set_ylabel('gamma')
    gamm_plt.set_xlabel('projection sizes')
    gamm_plt.loglog(np.array(wsizes), refere_gamma)
    gamm_plt.loglog(np.array(wsizes), approx_gamma, label=ds_name)


def savefig(name, postfix=''):
    plt.savefig('./plots/' + name + '_' + postfix + '.png')


subset_size = 500
cross_fold = 4
sigma = 1
ds_names = datasets.Datasets().getAllDatasetNames()
# ds_names = ['colic']
intermediate_sizes = 4

lin_clf = svm.LinearSVC(C=1, loss='hinge')
svm_clf = svm.SVC(kernel='rbf', C=1, gamma=sigma)

sets = datasets.Datasets()
projection_sizes = get_w_sizes(intermediate=intermediate_sizes)

for ds_name in [y for y in ds_names if y != 'magic04']:
    print ds_name
    ds = prepare_data(ds_name, subset_size)
    res_svm = run(svm_clf, ds, cross_fold)

    proj_runs = []
    for w in projection_sizes:
        proj_runs.append(run_approx(svm_clf, ds, cross_fold, w))
    print_res(res_svm, proj_runs, projection_sizes, ds_name, True)

plt.show()
savefig("svm_vs_proj", str(subset_size))