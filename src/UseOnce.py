# import numpy as np
# import scipy.stats as ss
# import matplotlib.pyplot as plt
#
# data_m = np.array([1, 2, 3, 4])  # (Means of your data)
# data_df = np.array([5, 6, 7, 8])  # (Degree-of-freedoms of your data)
# data_sd = np.array([11, 12, 12, 14])  # (Standard Deviations of your data)
#
# plt.errorbar([0, 1, 2, 3], data_m, yerr=ss.t.ppf(0.95, data_df) * data_sd)
# plt.xlim((-1, 4))
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
y = [1, 2, 3, 4, 5, 6, 7]
print y[0::2]

spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(1) * 10 + 100
flier_low = np.random.rand(1) * -10
data = np.concatenate((spread, center, flier_high, flier_low), 0)
data = np.random.randn(100)
print center

plt.boxplot(data, 0, 'gD', whis=[1, 99])
plt.show()

exit(0)

def fakeBootStrapper(n):
    '''
    This is just a placeholder for the user's method of
    bootstrapping the median and its confidence intervals.

    Returns an arbitrary median and confidence intervals
    packed into a tuple
    '''
    if n == 1:
        med = 0.1
        CI = (-0.25, 0.25)
    else:
        med = 0.2
        CI = (-0.35, 0.50)

    return med, CI


np.random.seed(2)
inc = 0.1
e1 = np.random.normal(0, 1, size=(500,))
e2 = np.random.normal(0, 1, size=(500,))
e3 = np.random.normal(0, 1 + inc, size=(500,))
e4 = np.random.normal(0, 1 + 2*inc, size=(500,))

treatments = [e1, e2, e3, e4]
med1, CI1 = fakeBootStrapper(1)
med2, CI2 = fakeBootStrapper(2)
medians = [None, None, med1, med2]
conf_intervals = [None, None, CI1, CI2]

fig, ax = plt.subplots()
pos = np.array(range(len(treatments))) + 1
bp = ax.boxplot(treatments, sym='k+', positions=pos,
                notch=1, bootstrap=500,
                usermedians=medians,
                conf_intervals=conf_intervals,
                whis=[5, 95])

ax.set_xlabel('treatment')
ax.set_ylabel('response')
plt.setp(bp['whiskers'], color='k', linestyle='-')
plt.setp(bp['fliers'], markersize=3.0)
plt.show()