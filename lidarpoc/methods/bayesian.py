import warnings
import numpy as np
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def optimize(objective, seed=0, **_):
    rng = np.random.default_rng(seed)
    while objective.remaining:
        candidates = rng.uniform(size=(256, objective.dim))
        if len(objective.history) < min(10, max(3, objective.budget // 4)):
            choice = candidates[0]
        else:
            x = np.asarray([row['unit'] for row in objective.history])
            y = np.asarray([row['loss'] for row in objective.history])
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(1e-5),
                                          normalize_y=True, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                gp.fit(x, y)
            mean, std = gp.predict(candidates, return_std=True)
            improvement = y.min() - mean - .001
            z = improvement / np.maximum(std, 1e-9)
            expected = improvement * ndtr(z) + std * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
            choice = candidates[expected.argmax()]
        objective(choice)
