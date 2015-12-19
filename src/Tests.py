import numpy as np
import time
from sklearn.svm import SVC
from numpy import random
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
import RKS_Sampler


def measure(func, arg, legend="time:"):
    start = time.clock()
    out = func(arg)
    end = time.clock()
    elapsed = end - start
    print legend, "\t", elapsed
    return out, elapsed


def measure_fit(cls, arg1, arg2, legend="time:"):
    start = time.clock()
    out = cls.fit(arg1, arg2)
    end = time.clock()
    print legend, "\t", end - start
    return out


def experiment(X, y, setup):
    setup['input_size'] = len(X)
    setup['result'] = {}

    sampler = RKS_Sampler.RKSSampler(1, n_dim=setup['rks_size'])
    proj = measure(sampler.transform, X, 'projection')
    setup['result']['rks_project_t'] = proj[1]

    start = time.clock()
    setup['rks_alg'].fit(proj[0], y)
    setup['result']['rks_train_t'] = time.clock() - start
    setup['result']['rks_score'] = setup['rks_alg'].score(proj[0], y)

    if not('ref_alg' in setup) or setup['ref_alg'] is None:
        return

    start = time.clock()
    setup['ref_alg'].fit(proj[0], y)
    setup['result']['ref_train_t'] = time.clock() - start
    setup['result']['ref_score'] = setup['ref_alg'].score(proj[0], y)


def print_result(setup):
    print '\n\rw size: %d; \t reference alg: %s' %(setup['rks_size'], type(setup['ref_alg']))
    print '\tRKS:      \tt:%.3f \t e:%.3f \t projection_t:%.3f' \
          % (setup['result']['rks_train_t'], setup['result']['rks_score'], setup['result']['rks_project_t'])

    if not('ref_alg' in setup) or setup['ref_alg'] is None:
        return

    print '\tReference:\tt:%.3f \t e:%.3f' \
          % (setup['result']['ref_train_t'], setup['result']['ref_score'])


reference_classifier = SGDClassifier()
rks_classifier = SGDClassifier()
exp = {'rks_alg': rks_classifier, 'ref_alg': reference_classifier, 'input_name': 'new', 'rks_size': 100}

X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
experiment(X, y, exp)

print_result(exp)



