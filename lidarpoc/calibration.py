"""RL and black-box calibration of bounded LiDAR realism parameters."""
import importlib

import numpy as np

from .eval.domain_metrics import describe, compare

NAMES = ('range_bias_m', 'range_noise_std_m', 'drop_probability', 'intensity_scale')
BOUNDS = np.array([[-.1, .1], [0., .1], [0., .5], [.5, 1.5]])
DEFAULT = np.array([0., 0., 0., 1.])


def encode(theta):
    return (np.asarray(theta) - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1)


def decode(unit):
    unit = np.asarray(unit, dtype=float)
    if unit.shape != (len(NAMES),) or not np.isfinite(unit).all():
        raise ValueError('Invalid calibration parameter vector')
    return BOUNDS[:, 0] + np.clip(unit, 0, 1) * np.ptp(BOUNDS, axis=1)


def apply(samples, theta, seed=0):
    rng, result = np.random.default_rng(seed), []
    bias, noise, drop, scale = decode(encode(theta))
    for sample in samples:
        valid = sample['valid'] & (rng.uniform(size=sample['valid'].shape) >= drop)
        offset = np.clip(bias + rng.normal(0, noise, size=valid.shape), -.2, .2)
        ranges = np.clip(sample['range'] + offset, .1, sample.get('max_range_m', 200.))
        result.append({**sample, 'range': np.where(valid, ranges, 0).astype('float32'), 'valid': valid,
                       'intensity': np.where(valid, np.clip(sample['intensity'] * scale, 0, 1), 0).astype('float32')})
    return result


class Objective:
    def __init__(self, real, generated, max_range, budget, seed=0):
        self.real, self.generated, self.max_range = describe(real), generated, max_range
        self.budget, self.seed, self.history = budget, seed, []
        self.dim, self.initial = len(NAMES), encode(DEFAULT)
        self.last_components = {}

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def observation(self, unit):
        row = self.last_components
        return np.r_[unit, [row['loss'], row['range_wasserstein_normalized'], row['return_rate_mae'],
                            row['intensity_wasserstein'] or 0., row['temporal_discrepancy'] or 0.]].astype('float32')

    def __call__(self, unit):
        if not self.remaining:
            raise RuntimeError('Calibration evaluation budget exhausted')
        theta = decode(unit)
        self.last_components = compare(self.real, describe(apply(self.generated, theta, self.seed)), self.max_range)
        self.history.append({'evaluation': len(self.history) + 1, 'unit': np.clip(unit, 0, 1).tolist(), **self.last_components})
        return self.last_components['loss']


def calibrate(method, real, generated, validation_real, validation_generated, max_range, budget, seed=0, device='cpu'):
    objective = Objective(real, generated, max_range, budget, seed)
    objective(objective.initial)
    policy = importlib.import_module(f'lidarpoc.methods.{method}').optimize(objective, seed=seed, device=device)
    # Separate complete traces for reward fitting and checkpoint selection.
    validation = []
    target = describe(validation_real)
    for row in sorted(objective.history, key=lambda row: row['loss'])[:min(5, budget)]:
        score = compare(target, describe(apply(validation_generated, decode(row['unit']), seed + 1000)), max_range)
        validation.append({'unit': row['unit'], **score})
    selected = min(validation, key=lambda row: row['loss'])
    return {'theta': decode(selected['unit']).tolist(), 'validation': selected,
            'history': objective.history, 'policy': policy, 'evaluations': len(objective.history)}
