import TestKernelApproximation as ka
import TestPerformanceVsSVM as svma
import matplotlib.pyplot as plt
import settings

#Program main. Runs all tests.

doItFast = False

settings.init(doItFast)

ka.run_all_relevant(faste=doItFast)
plt.figure()
svma.run_relevant_tests(fast=doItFast)

