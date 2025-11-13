# surrogate/bnn_stub.py
# Placeholder Bayesian Neural Network interface (requires PyTorch / pyro / tensorflow-probability for full impl).
import numpy as np

class BNNStub:
    def __init__(self):
        pass

    def fit(self, X, y):
        # Placeholder: store empirical mean/std
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y)

    def predict(self, X):
        # Return mean prediction and a simple uncertainty estimate
        mean = np.full((len(X),), self.y_mean)
        std = np.full((len(X),), np.std(self.y_mean) + 1e-6)
        return mean, std