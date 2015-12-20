import datasets
import numpy as np
import time
from sklearn import svm
from sklearn import cross_validation
from RKS_Sampler import RKSSampler
from matplotlib import pyplot as plt

# [ 'breast-w', 'colic',    'credit-a', 'credit-g', 'heart-statlog',    'hepatitis',    'magic04',  'sick',    'vote']
#   (699, 9),   (368, 7),   (690, 6),   (1000, 7),  (270, 13),          (155, 19),      (19019, 9), (3772, 7), (435, 16


def get_w_sizes(min_v=10, max_v=1000, intermediate=4):
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


def print_res(res_svm, proj_runs, wsizes):
    f, (time_plt, erro_plt, gamm_plt) = plt.subplots(3, sharex='col')

    time_plt.set_title('time')
    time_plt.loglog(np.array(wsizes), np.array([res_svm[0]]*len(wsizes)))
    time_plt.loglog(np.array(wsizes), np.array([x[0] for x in proj_runs]))

    erro_plt.set_title('error')
    erro_plt.semilogx(np.array(wsizes), np.array([res_svm[1]]*len(wsizes)))
    erro_plt.semilogx(np.array(wsizes), np.array([x[1] for x in proj_runs]))

    gamm_plt.set_title('gamma')
    gamm_plt.semilogx(np.array(wsizes), np.array([res_svm[2]]*len(wsizes)))
    gamm_plt.semilogx(np.array(wsizes), np.array([x[2] for x in proj_runs]))

    plt.show()


subset_size = 200
cross_fold = 4
sigma = 1
ds_names = datasets.Datasets().getAllDatasetNames()
ds_names = ['colic']

lin_clf = svm.LinearSVC(C=1, loss='hinge')
svm_clf = svm.SVC(kernel='rbf', C=1, gamma=sigma)

sets = datasets.Datasets()
projection_sizes = get_w_sizes()
# for ds_name in :
for ds_name in ds_names:
    ds = prepare_data(ds_name, subset_size)
    res_svm = run(svm_clf, ds, cross_fold)

    proj_runs = []
    for w in projection_sizes:
        proj_runs.append(run_approx(svm_clf, ds, cross_fold, w))
    print_res(res_svm, proj_runs, projection_sizes)