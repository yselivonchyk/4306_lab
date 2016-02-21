import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

err = [x2*.8, x2*1.1]
print err
err = err - x2
print err
err = np.absolute(err)

plt.errorbar(x1, x2, yerr=err, fmt='o')


plt.show()