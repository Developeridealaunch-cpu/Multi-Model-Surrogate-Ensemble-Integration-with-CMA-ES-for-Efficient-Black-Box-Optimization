"""
Gaussian Process surrogate wrapper.
"""
from __future__ import annotations
import numpy as np
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
except Exception:
    GaussianProcessRegressor = None
    Matern = WhiteKernel = C = None

class GPModel:
    def __init__(self, input_dim:int, normalize_y:bool=True, random_state:int|None=None):
        self.input_dim = input_dim
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.model = None
        if GaussianProcessRegressor is not None:
            kernel = C(1.0, (1e-6, 1e6)) * Matern(length_scale=np.ones(input_dim), nu=2.5) + WhiteKernel(noise_level=1e-6)
            self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, random_state=random_state, n_restarts_optimizer=1)

    def fit(self, X:np.ndarray, y:np.ndarray):
        if self.model is None:
            raise RuntimeError("scikit-learn not available for GPModel.")
        self.model.fit(np.asarray(X), np.asarray(y).reshape(-1, 1))

    def predict(self, X:np.ndarray, return_std:bool=True):
        if self.model is None:
            raise RuntimeError("scikit-learn not available for GPModel.")
        mu, std = self.model.predict(np.asarray(X), return_std=True)
        return mu.reshape(-1), std.reshape(-1)

    def is_ready(self)->bool:
        return self.model is not None
