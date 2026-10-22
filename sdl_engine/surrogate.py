"""
surrogate.py - the surrogate model.

Term: surrogate model
    A cheap model that stands in for an expensive experiment. In a self-driving
    lab it is the thing the "what next" decision is based on.

What this file must provide:
    A prediction (mu) AND an uncertainty (sigma). Without sigma there is no way
    to explore, so a model that only predicts is not usable here.

Two options:
    RandomForestSurrogate : disagreement among 100 trees is the uncertainty.
                            Robust on one-hot categorical data and fast. Default.
    GPSurrogate           : Gaussian process. Strong on continuous inputs with
                            few data points. The model used in the course.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


class RandomForestSurrogate:
    """Random forest.

    Where the uncertainty comes from:
        Each of the 100 trees makes its own prediction. When they disagree, the
        model is unsure. The spread (standard deviation) of those predictions is
        used as sigma. It costs almost nothing extra to compute, which is why
        uncertainty is often described as free here.

    Limitation:
        Outside the range of the training data its predictions collapse toward
        the mean without saying so. This is worse than in a GP.
    """

    name = "RandomForest"

    def __init__(self, n_estimators=100, random_state=0, n_jobs=1, **kw):
        # n_jobs=1 by default: inside the loop there are only tens to hundreds of
        # training rows, so thread setup costs more than the work itself.
        # Measured here, n_jobs=-1 was about 1.5x slower.
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            **kw,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Return (mu, sigma)."""
        per_tree = np.stack([t.predict(X) for t in self.model.estimators_])
        mu = per_tree.mean(axis=0)
        sigma = per_tree.std(axis=0)
        # A sigma of exactly zero would divide by zero inside EI.
        sigma = np.maximum(sigma, 1e-9)
        return mu, sigma


class GPSurrogate:
    """Gaussian process (the model used in the course exercises).

    The kernel measures how similar two settings are, and that similarity fills
    in the values between measured points. Uncertainty is narrow near measured
    points and widens away from them.

    Careful:
        It slows down sharply past a few thousand candidates (cost grows with the
        cube of the number of observations). On one-hot categorical inputs the
        notion of distance is weak, so it can do worse than the forest.
    """

    name = "GP"

    def __init__(self, random_state=0, noise_level=1e-2):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(noise_level, (1e-8, 1e1)))
        self.model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=random_state,
            n_restarts_optimizer=1,
        )
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        mu, sigma = self.model.predict(self.scaler.transform(X), return_std=True)
        return mu, np.maximum(sigma, 1e-9)


SURROGATES = {
    "rf": RandomForestSurrogate,
    "gp": GPSurrogate,
}


def build_surrogate(kind="rf", random_state=0, **kw):
    if kind not in SURROGATES:
        raise KeyError(f"Unknown surrogate: {kind}. Available: {list(SURROGATES)}")
    return SURROGATES[kind](random_state=random_state, **kw)
