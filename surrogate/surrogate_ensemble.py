"""
Multi-model surrogate ensemble with GP, SVR, RBF, Polynomial, and MC-Dropout net.
Provides:
- predict(): mean, std, and per-model matrix P
- rank_scores(): rank-based scores
- gating_weights(): Kendall-τ weights per model
- novelty_signals(): ensemble disagreement, rank stability, model diversity (for "novelty" comparison)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from .gp_model import GPModel

try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from sklearn.metrics import pairwise_distances
    import scipy.stats as st
except Exception:
    SVR = PolynomialFeatures = make_pipeline = Ridge = pairwise_distances = None
    st = None

# Lightweight RBF regressor
class RBFRegressor:
    def __init__(self, gamma:float=1.0, ridge:float=1e-6):
        self.gamma = gamma
        self.ridge = ridge
        self.X = None
        self.w = None

    def fit(self, X, y):
        if pairwise_distances is None:
            raise RuntimeError("scikit-learn required for RBFRegressor")
        self.X = np.asarray(X)
        D = pairwise_distances(self.X, self.X, metric="euclidean")
        K = np.exp(-self.gamma * D**2) + self.ridge*np.eye(len(self.X))
        self.w = np.linalg.solve(K, y)

    def predict(self, X):
        if pairwise_distances is None:
            raise RuntimeError("scikit-learn required for RBFRegressor")
        X = np.asarray(X)
        D = pairwise_distances(X, self.X, metric="euclidean")
        K = np.exp(-self.gamma * D**2)
        return K @ self.w

# MC-Dropout net (BNN-like)
try:
    import torch, torch.nn as nn, torch.nn.functional as F
except Exception:
    torch = nn = F = None

class MCDropoutNet(nn.Module if nn else object):
    def __init__(self, input_dim:int, hidden:int=64, p:float=0.1):
        if nn is None:
            raise RuntimeError("PyTorch not available for MCDropoutNet")
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

@dataclass
class SurrogateEnsemble:
    input_dim:int
    n_models:int=5
    random_state:int|None=None
    models:list=field(default_factory=list)
    X_train:np.ndarray|None=None
    y_train:np.ndarray|None=None
    use_neural:bool=True

    def __post_init__(self):
        self.models = []
        # GP
        self.gp = GPModel(self.input_dim, normalize_y=True, random_state=self.random_state)
        if self.gp.is_ready():
            self.models.append(("gp", self.gp))
        # SVR
        if SVR is not None:
            self.models.append(("svr", SVR(C=10.0, epsilon=0.01, kernel="rbf", gamma="scale")))
        # Polynomial
        if PolynomialFeatures is not None:
            self.models.append(("poly", make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))))
        # RBF
        if pairwise_distances is not None:
            self.models.append(("rbf", RBFRegressor(gamma=1.0/max(1,self.input_dim))))
        # MC-Dropout
        self.mcnet = None
        if self.use_neural and (torch is not None):
            self.mcnet = MCDropoutNet(self.input_dim)
            self.optim = torch.optim.Adam(self.mcnet.parameters(), lr=1e-3)

    def fit(self, X, y, epochs:int=80):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.X_train, self.y_train = X, y
        for name, m in self.models:
            if name == "gp":
                m.fit(X, y)
            elif name in ("svr","poly","rbf"):
                m.fit(X, y)
        if self.mcnet is not None and len(X) >= 8:
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32)
            self.mcnet.train()
            for _ in range(min(epochs, 80)):
                self.optim.zero_grad()
                pred = self.mcnet(Xt)
                loss = ((pred - yt)**2).mean()
                loss.backward()
                self.optim.step()

    def predict(self, X, n_mc:int=20):
        X = np.asarray(X, dtype=float)
        preds = []
        for name, m in self.models:
            if name == "gp":
                mu, _ = m.predict(X, return_std=True)
                preds.append(mu.reshape(-1))
            elif name in ("svr","poly","rbf"):
                mu = m.predict(X).reshape(-1)
                preds.append(mu)
        if self.mcnet is not None:
            self.mcnet.train()
            with torch.no_grad():
                Xt = torch.tensor(X, dtype=torch.float32)
                outs = []
                for _ in range(n_mc):
                    outs.append(self.mcnet(Xt).cpu().numpy().reshape(-1))
                preds.append(np.mean(outs, axis=0))
        if len(preds)==0:
            raise RuntimeError("No surrogate models available")
        P = np.vstack(preds)  # [n_models, n_points]
        mean = P.mean(axis=0)
        std = P.std(axis=0)
        return mean, std, P

    def rank_scores(self, X):
        mu, _, _ = self.predict(X)
        order = np.argsort(mu)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(mu))
        return ranks.astype(float)

    def gating_weights(self, X_eval, y_eval):
        if st is None:
            return np.ones(len(self.models))/max(1,len(self.models))
        _, _, P = self.predict(X_eval)
        weights = []
        for i in range(P.shape[0]):
            tau, _ = st.kendalltau(P[i], y_eval)
            tau = 0.0 if np.isnan(tau) else max(tau, 0.0)
            weights.append(tau + 1e-6)
        w = np.asarray(weights)
        w /= w.sum()
        return w

    def novelty_signals(self, X):
        """
        Return a dict with ensemble disagreement, rank stability, diversity.
        - disagreement: mean predictive std across points (ensemble variance)
        - rank stability: average Kendall-τ between model rankings (higher => more stable)
        - diversity: average pairwise L2 distance between model predictions (scaled)
        """
        _, _, P = self.predict(X)
        # disagreement
        disagreement = P.std(axis=0).mean()
        # rank stability
        if P.shape[0] >= 2:
            ranks = np.argsort(P, axis=1)
            import itertools
            taus = []
            for i,j in itertools.combinations(range(P.shape[0]), 2):
                # compute tau on predictions directly
                import scipy.stats as st2
                tau, _ = st2.kendalltau(P[i], P[j])
                if not np.isnan(tau):
                    taus.append(tau)
            rank_stability = float(np.mean(taus)) if len(taus) else 0.0
        else:
            rank_stability = 0.0
        # diversity
        D = 0.0
        for i in range(P.shape[0]):
            for j in range(i+1, P.shape[0]):
                D += np.linalg.norm(P[i]-P[j]) / (1+np.linalg.norm(P[i])+np.linalg.norm(P[j]))
        pairs = P.shape[0]*(P.shape[0]-1)/2
        diversity = float(D / max(1, pairs))
        return {"disagreement": float(disagreement), "rank_stability": rank_stability, "diversity": diversity}
