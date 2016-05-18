import TestKernelApproximation as ka
import TestPerformanceVsSVM as svma
import matplotlib.pyplot as plt

#Program main. Runs all tests.

# ka.run_all_relevant()
# svma.run_relevant_tests()

# ka.run_all_relevant(faste=True)
# plt.show()
svma.run_relevant_tests(fast=True)
plt.show()
