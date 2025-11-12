import numpy as np
def rastrigin(x):
    x = np.asarray(x); return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

