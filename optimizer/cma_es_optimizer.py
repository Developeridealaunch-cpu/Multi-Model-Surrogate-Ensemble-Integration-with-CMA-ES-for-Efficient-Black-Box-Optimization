"""
Surrogate-assisted CMA-ES with ESR, DAE-SMC, MSES variants.
Includes: LHS sampling, normalization to bounds, simple covariance update, and history tracking.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .acquisition import expected_improvement, ucb

try:
    from scipy.stats import qmc
except Exception:
    qmc = None

def lhs_samples(bounds, n, seed=None):
    dim = len(bounds)
    if qmc is None:
        rng = np.random.RandomState(seed)
        return np.array([rng.uniform([b[0] for b in bounds],[b[1] for b in bounds]) for _ in range(n)])
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    X = sampler.random(n)
    l = np.array([b[0] for b in bounds]); u = np.array([b[1] for b in bounds])
    return qmc.scale(X, l, u)

def denormalize(Z, bounds):
    l = np.array([b[0] for b in bounds]); u = np.array([b[1] for b in bounds])
    return Z * (u - l) + l

@dataclass
class CMAESState:
    mean: np.ndarray
    sigma: float
    C: np.ndarray

def cmaes_init(dim, x0=None, sigma0=0.3):
    mean = np.zeros(dim) if x0 is None else np.asarray(x0, dtype=float)
    C = np.eye(dim)
    return CMAESState(mean, sigma0, C)

def cmaes_sample(state:CMAESState, lam:int, rng):
    A = np.linalg.cholesky(state.C + 1e-9*np.eye(len(state.mean)))
    z = rng.randn(lam, len(state.mean))
    y = z @ A.T
    x = state.mean + state.sigma * y
    return x, y

def cmaes_update(state:CMAESState, y_sel, weights):
    m_old = state.mean.copy()
    state.mean = m_old + state.sigma * (weights @ y_sel)
    C_new = 0.9 * state.C + 0.1 * (y_sel.T @ np.diag(weights) @ y_sel)
    state.C = 0.5*(C_new + C_new.T) + 1e-9*np.eye(len(state.mean))
    state.sigma *= 0.99

class CMAESOptimizer:
    def __init__(self, dim, bounds, surrogate, max_evals=200, seed=42, variant="ESR", batch_size=8):
        self.dim = dim
        self.bounds = bounds
        self.surr = surrogate
        self.max_evals = max_evals
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.variant = variant  # "ESR", "DAE-SMC", "MSES"
        self.batch_size = batch_size
        self.history = []  # (evals, best_y)

    def optimize(self, f, verbose=True, init_points=20):
        X0 = lhs_samples(self.bounds, init_points, seed=self.seed)
        y0 = np.array([f(x) for x in X0])
        evals = len(y0)
        best_idx = np.argmin(y0)
        best_x, best_y = X0[best_idx], y0[best_idx]

        self.surr.fit(X0, y0)
        state = cmaes_init(self.dim, x0=np.clip(best_x, [b[0] for b in self.bounds], [b[1] for b in self.bounds]), sigma0=0.6)
        lam = max(6, 2 * self.dim)
        weights = np.ones(lam) / lam
        self.history = [(evals, float(best_y))]

        while evals < self.max_evals:
            Xcand_n, Ycand = cmaes_sample(state, lam, self.rng)
            Z = 0.5*(np.tanh(Xcand_n/3.0)+1.0)  # [-inf,inf] -> [0,1]
            Xcand_box = denormalize(Z, self.bounds)

            mu, std, P = self.surr.predict(Xcand_box)

            if self.variant.upper() == "ESR":
                order = np.argsort(mu)  # rank by mean
            elif self.variant.upper() == "DAE-SMC":
                acq = ucb(mu, std, kappa=1.5)
                order = np.argsort(acq)
            else:  # MSES: mix
                acq_ei = expected_improvement(mu, std, best_y, xi=0.01)
                acq_ucb = ucb(mu, std, kappa=1.0)
                acq = 0.5*acq_ucb + 0.5*(-acq_ei)
                order = np.argsort(acq)

            topk_idx = order[: self.batch_size]
            X_eval = Xcand_box[topk_idx]
            y_eval = np.array([f(x) for x in X_eval])
            evals += len(y_eval)

            # Update best
            idx = np.argmin(y_eval)
            if y_eval[idx] < best_y:
                best_y = float(y_eval[idx])
                best_x = X_eval[idx].copy()

            # Refit surrogate
            X0 = np.vstack([X0, X_eval])
            y0 = np.concatenate([y0, y_eval])
            self.surr.fit(X0, y0, epochs=50)

            # CMA-ES update
            y_sel = Xcand_n[topk_idx]
            cmaes_update(state, y_sel, weights[:len(topk_idx)])

            self.history.append((evals, float(best_y)))
            if verbose:
                print(f"[{self.variant}] evals={evals:4d}  best={best_y:.6f}")

        return {"best_x": best_x, "best_y": float(best_y), "history": self.history}
