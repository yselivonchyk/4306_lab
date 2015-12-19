import numpy as np


def distance(x, y, delta=1):
    norm = np.linalg.norm(x-y)
    return np.exp(-0.5 * np.square(norm)/np.square(delta))
