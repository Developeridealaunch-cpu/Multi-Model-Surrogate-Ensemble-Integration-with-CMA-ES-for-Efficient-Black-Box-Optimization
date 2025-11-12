"""
Acquisition functions: EI and UCB.
"""
import numpy as np
from scipy.stats import norm

def expected_improvement(mu, sigma, best_f, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    imp = best_f - mu - xi
    Z = imp / sigma
    return (imp * norm.cdf(Z) + sigma * norm.pdf(Z))

def ucb(mu, sigma, kappa=2.0):
    return mu - kappa * sigma  # minimization
