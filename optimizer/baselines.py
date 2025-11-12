"""
Baselines: pure CMA-ES (very simple) and surrogate-assisted helper.
"""
import numpy as np
from .cma_es_optimizer import CMAESOptimizer

def pure_cmaes(f, dim, bounds=None, max_evals=200, seed=0):
    if bounds is None:
        bounds = [(-5,5)]*dim
    rng = np.random.RandomState(seed)
    x = rng.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    sigma = 0.5
    best_x, best_y = x.copy(), f(x)
    evals = 1
    while evals < max_evals:
        xs = x + sigma * rng.randn(10, dim)
        xs = np.clip(xs, [b[0] for b in bounds], [b[1] for b in bounds])
        ys = np.array([f(xi) for xi in xs])
        evals += len(xs)
        i = np.argmin(ys)
        if ys[i] < best_y:
            best_y, best_x = ys[i], xs[i]
            x = xs[i]
        sigma *= 0.99
    return best_x, float(best_y)
